# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Function to train the rendering model."""

import functools
import json
import os
from typing import Any, Callable, Tuple

from absl import logging

from clu import metric_writers
from clu import metrics
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import flax.linen as nn
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf

from gen_patch_neural_rendering.src import datasets
from gen_patch_neural_rendering.src import models
from gen_patch_neural_rendering.src.utils import data_types
from gen_patch_neural_rendering.src.utils import file_utils
from gen_patch_neural_rendering.src.utils import model_utils
from gen_patch_neural_rendering.src.utils import render_utils
from gen_patch_neural_rendering.src.utils import train_utils

###################################################################################################################

def train_step(
    model, rng, state,
    batch, learning_rate_fn,
    weight_decay,
    config):
  """Perform a single train step.

  Args:
    model: Flax module for the model. The apply method must take input images
      and a boolean argument indicating whether to use training or inference
      mode.
    rng: random number generator.
    state: State of the model (optimizer and state).
    batch: Training inputs for this step.
    learning_rate_fn: Function that computes the learning rate given the step
      number.
    weight_decay: Weighs L2 regularization term.
    config: experiment config dict.

  Returns:
    The new model state and dictionary with metrics.
  """
  logging.info("train_step(batch=%s)", batch)

  step = state.step + 1
  lr = learning_rate_fn(step)

  #Splitting the random number generator is useful in various scenarios,
  #such as when you need to generate multiple sources of randomness in parallel computations
  #or when you want to ensure reproducibility by using different random number streams for different parts of your code.
  rng, key_0, key_1 = jax.random.split(rng, 3)

#####################################################################################################################
  #computes the loss and other statistics for a given set of model parameters.
  def loss_fn(params):

    """
    Compute the loss and statistics for a given set of model parameters.

    Args:
        params: Model parameters.

    Returns:
        The total loss and computed statistics.
    """

    variables = {"params": params}

    ret = model.apply(
        variables, key_0, key_1, batch, randomized=config.model.randomized)

    #This condition checks if the ret list contains either one set of output (coarse only) or two sets of output (coarse as ret[0] and fine as ret[1]).
    #If the length of ret is not 1 or 2, a ValueError is raised.
    if len(ret) not in (1, 2):
      raise ValueError(
          "ret should contain either 1 set of output (coarse only), or 2 sets"
          "of output (coarse as ret[0] and fine as ret[1]).")
    #------------------------------------------------------------------------
    # Main prediction
    # The main prediction is always at the end of the ret list.

    #These lines extract the RGB output from ret[-1], which represents the main prediction.
    rgb, unused_disp, unused_acc = ret[-1]
    batch_pixels = model_utils.uint2float(batch.target_view.rgb)

    #computes the loss as the mean squared difference between the predicted RGB values (rgb)
    #and the target RGB values (batch_pixels[Ellipsis, :3]).
    loss = ((rgb - batch_pixels[Ellipsis, :3])**2).mean()

    #The PSNR (Peak Signal-to-Noise Ratio) is computed
    psnr = model_utils.compute_psnr(loss)

    #------------------------------------------------------------------------
    # Coarse / Regularization Prediction

    #If the length of ret is greater than 1, it means there are both coarse and fine predictions.
    #In that case, the function computes the loss and PSNR for the coarse prediction (ret[0]).
    #If there is only one set of output, the loss and PSNR for the coarse prediction are set to 0.
    if len(ret) > 1:
      # If there are both coarse and fine predictions, we compute the loss for
      # the coarse prediction (ret[0]) as well.
      rgb_c, unused_disp_c, unused_acc_c = ret[0]
      loss_c = ((rgb_c - batch_pixels[Ellipsis, :3])**2).mean()
      psnr_c = model_utils.compute_psnr(loss_c)
    else:
      loss_c = 0.
      psnr_c = 0.

    #------------------------------------------------------------------------
    # Weight Regularization
    weight_penalty_params = jax.tree_leaves(variables["params"])
    weight_l2 = sum(
        [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2

    #------------------------------------------------------------------------
    # Compute total loss and wrap the stats

    #The total_loss is the sum of the main loss, coarse loss, and the weight penalty
    total_loss = loss + loss_c + weight_penalty

    #The function creates a Stats object using train_utils.Stats to store the loss and PSNR values for both the main prediction and the coarse prediction,
    #along with the weight L2 norm. Finally, the total_loss and stats are returned as a tuple.
    stats = train_utils.Stats(
        loss=loss, psnr=psnr, loss_c=loss_c, psnr_c=psnr_c, weight_l2=weight_l2)
    return total_loss, stats

#######################################################################################################################
  #------------------------------------------------------------------------
  # Compute Graidents

  #has_aux=True-Argument gibt an, dass die loss_fn auch zusätzliche Statistiken zurückgibt.
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, stats), grad = grad_fn(state.params)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")

  #------------------------------------------------------------------------
  # Update States
  new_state = state.apply_gradients(grads=grad)

  metrics_update = train_utils.TrainMetrics.gather_from_model_output(
      total_loss=loss,
      loss=stats.loss,
      psnr=stats.psnr,
      loss_c=stats.loss_c, #coarse loss/ grober Verlust
      psnr_c=stats.psnr_c, #coarse psnr/ grober psnr
      weight_l2=stats.weight_l2,
      learning_rate=lr)
  return new_state, metrics_update, rng

#######################################################################################################################
def eval_step(state, rng, batch,
              render_pfn, config):
  """Compute the metrics for the given model in inference mode.

  The model is applied to the inputs with train=False using all devices on the
  host. Afterwards metrics are averaged across *all* devices (of all hosts).
  Args:
    state: Replicate model state.
    rng: random number generator.
    batch: data_types.Batch. Inputs that should be evaluated.
    render_pfn: pmaped render function.
    config: exepriment config.

  Returns:
    Dictionary of the replicated metrics.
  """
  logging.info("eval_step=================")
  #creating a dictionary variables that contains the model parameters (params) extracted from the replicated state
  #The jax.tree_map function is used to apply a lambda function that selects the first element of each parameter tuple.
  #The jax.device_get function ensures that the parameters are retrieved from the correct device.
  variables = {
      "params": jax.device_get(jax.tree_map(lambda x: x[0], state)).params,
  }

  pred_color, pred_disp, pred_acc = render_utils.render_image(
      functools.partial(render_pfn, variables),
      batch,
      rng,
      render_utils.normalize_disp(config.dataset.name),
      chunk=config.eval.chunk)

  return pred_color, pred_disp, pred_acc #disparity/Ungleichheit

#######################################################################################################################
def train_and_evaluate(config, workdir):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  #"#Schritt 1: Überprüfen der Batch-Größe auf Teilbarkeit durch die Anzahl der Geräte."

  #Um eine effiziente Verteilung sicherzustellen, muss die Batch-Größe gleichmäßig auf die Geräte aufgeteilt werden können.
  if config.dataset.batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")

  ##################################################################################################################
 # "#Schritt 2: Erstellen des Arbeitsverzeichnisses, falls es nicht vorhanden ist."
  tf.io.gfile.makedirs(workdir)

#  "#Schritt 3: Festlegen des Zufallszahlengenerator-Saatguts für deterministisches Training.#
  # Deterministic training.
  #(config.seed) wird verwendet, um die Reproduzierbarkeit des Trainings zu gewährleisten.
  #Der Seed wird in einen PRNG-Schlüssel (rng) umgewandelt.
  rng = jax.random.PRNGKey(config.seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded
  # by different hosts
  np.random.seed(20201473 + jax.process_index())

  #----------------------------------------------------------------------------
  # Build input pipeline.
  #'#Schritt 4: Aufbau der Eingabepipeline und Vorbereitung der Trainings- und Evaluierungsdatensätze.'
  #Dann wird ein zweiter Zufallsgenerator (data_rng) aus dem ersten Zufallsgenerator (rng) abgeleitet und für die Datenmanipulation verwendet.
  #Dieser Zufallsgenerator wird auch an den aktuellen Prozessindex angepasst, um die Daten zwischen den Hosts zu mischen.
  rng, data_rng = jax.random.split(rng)
  data_rng = jax.random.fold_in(data_rng, jax.process_index())

  #Anschließend wird eine Liste von Szenenpfaden (scene_path_list) mit Hilfe von Hilfsfunktionen erstellt. D
  #Diese Pfade werden für die Erstellung des Trainingsdatensatzes verwendet.
  scene_path_list = train_utils.get_train_scene_list(config)
  print("scene path list: ", scene_path_list[0])
  train_ds = datasets.create_train_dataset(config, scene_path_list[0])
  print("created train_ds")

  #config.dataset.render_style == "neu"

  _, eval_ds_dict = datasets.create_eval_dataset(config)
  print("created eval_ds")
  _, eval_ds = eval_ds_dict.popitem()
  example_batch = train_ds.peek()

  #Schritte 1 bis 4: legt somit den Grundstein für die Datenverarbeitung, indem das Arbeitsverzeichnis erstellt, die Zufallsgeneratoren initialisiert und die Trainings- und Evaluationsdatensätze erstellt werden.
  #Diese Daten werden später im Trainings- und Evaluierungsschleifen verwendet, um das Modell zu trainieren und zu bewerten.

  ##################################################################################################################
  #----------------------------------------------------------------------------
  "#Schritt 5: Berechnen der Anzahl der Trainingsschritte basierend auf der Konfiguration."
  print("STEP 5")
  # Learning rate schedule.
  num_train_steps = config.train.max_steps
  #Wenn config.train.max_steps den Wert -1 hat, bedeutet dies, dass keine spezifische Anzahl von Trainingsschritten festgelegt wurde.
  #In diesem Fall wird num_train_steps auf die Größe des Trainingsdatensatzes (train_ds.size()) gesetzt.
  if num_train_steps == -1:
    num_train_steps = train_ds.size()
  steps_per_epoch = num_train_steps // config.train.num_epochs
  logging.info("num_train_steps=%d, steps_per_epoch=%d", num_train_steps,
               steps_per_epoch)

  "#Schritt 6: Erstellen des Lernratenplans."
  learning_rate_fn = train_utils.create_learning_rate_fn(config)

  #----------------------------------------------------------------------------
  "#Schritt 7: Initialisieren des Modells und Laden von Checkpoints, falls vorhanden."
  print("STEP 7")
  # Initialize model.
  rng, model_rng = jax.random.split(rng)
  model, state = models.create_train_state(
      config,
      model_rng,
      learning_rate_fn=learning_rate_fn,
      example_batch=example_batch,
  )

  #----------------------------------------------------------------------------
  # Set up checkpointing of the model and the input pipeline.

  # check if the job was stopped and relaunced
  latest_ckpt = checkpoints.latest_checkpoint(workdir)
  if latest_ckpt is None:
    # No previous checkpoint. Then check for pretrained weights.
    if config.train.pretrain_dir:
      state = checkpoints.restore_checkpoint(config.train.pretrain_dir, state)
  else:
    state = checkpoints.restore_checkpoint(workdir, state)

  initial_step = int(state.step) + 1
  step_per_scene = config.train.switch_scene_iter#wie oft eine Szene gewechselt wird.
  if config.dev_run:
    print("config.dev_run==TRUE")
    jnp.set_printoptions(precision=2)
    np.set_printoptions(precision=2)
    step_per_scene = 3

  #----------------------------------------------------------------------------
  "#Schritt 8: Verteilung des Trainings auf mehrere Geräte."
  print("STEP 8")
  # Distribute training.
  #Zunächst wird der Trainingszustand state mit Hilfe der Funktion flax_utils.replicate() auf alle verfügbaren Geräte repliziert.
  #Dadurch wird der Zustand auf jedes Gerät dupliziert, um das Training parallel auf mehreren Geräten durchführen zu können.
  state = flax_utils.replicate(state)
  #Anschließend wird die Funktion train_step() teilweise angewendet
  #und für die parallele Ausführung über mehrere Geräte mit jax.pmap() parallelisiert.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          model=model,
          learning_rate_fn=learning_rate_fn,
          weight_decay=config.train.weight_decay,
          config=config,
      ),
      axis_name="batch",
  )
  #Die Funktion p_train_step stellt nun eine parallelisierte Version des train_step()-Funktionsaufrufs dar,
  #die auf allen replizierten Trainingszuständen gleichzeitig ausgeführt werden kann.

  #Durch diese Verteilung des Trainings auf mehrere Geräte wird die Rechenleistung effizient genutzt und das Training beschleunigt.
  #Jeder Geräte-Thread kann unabhängig von den anderen Threads auf seinem spezifischen Teil der Daten arbeiten
  #und die Ergebnisse werden anschließend synchronisiert.

######################################################################################################################

  "#Schritt 9: Abrufen der Rendering-Funktion für die Evaluation."
  print("STEP 9")
  # Get distributed rendering function
  render_pfn = render_utils.get_render_function(
      model=model,
      config=config,
      randomized=False,  # No randomization for evaluation.
  )
  #Die Rendering-Funktion ist verantwortlich für die Generierung von Farb- und Tiefenkarten basierend auf den Eingabedaten und dem trainierten Modell.
  #Sie nutzt die gelernten Parameter des Modells, um eine Vorhersage für die Darstellung des Szeneninhalts zu treffen.
  #----------------------------------------------------------------------------

######################################################################################################################

  train_loss_list = []
  train_psnr_list = []
  weight_l2_list = []
  learining_rate_list = []

  total_loss_list = []
  train_loss_std_list = []
  train_loss_c_list = []
  train_psnr_c_list = []

  "#Schritt 10: Erstellen von Metrik-Schreibern für das Logging."
  # Prepare Metric Writers
  writer = metric_writers.create_default_writer(
      #The just_logging parameter is set to True if the process index (jax.process_index()) is greater than 0,
      #indicating that this writer is only used for logging and not for writing metrics.
      workdir, just_logging=jax.process_index() > 0)
  #Additionally, it checks if the initial step (initial_step) is equal to 1.
  #If this condition is true, the experiment's hyperparameters (config) are written to the metric writer using writer.write_hparams().
  #This allows storing and logging the experiment's configuration for future reference.
  if initial_step == 1:
    writer.write_hparams(dict(config))

######################################################################################################################

  "#Schritt 11: Starten der Trainingsschleife."
  print("Step 11")
  logging.info("Starting training loop at step %d.", initial_step)
  hooks = []
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=num_train_steps, writer=writer)
  #This ensures that progress reporting is performed only by the first process, while other processes do not execute this action.
  if jax.process_index() == 0:
    hooks += [
        report_progress,
    ]
  train_metrics = None

  ######################################################################################################################

  "#Schritt 12: Vorbereiten des Trainingsdatensatzes für die Geräte."
  print("Step 12")
  # Prefetch_buffer_size = 6 x batch_size
  ptrain_ds = flax.jax_utils.prefetch_to_device(train_ds, 6)#helps to improve training performance by overlapping data loading and computation.
  n_local_devices = jax.local_device_count()#This indicates the number of devices available for training.
  rng = rng + jax.process_index()  # Make random seed separate across hosts.
  keys = jax.random.split(rng, n_local_devices)  # For pmapping RNG keys.

#######################################################################################################################

  "#Schritt 13: Schleife über die Trainings-Steps."
  print("Step13")

  #with statement to ensure that the metric writer's buffers are flushed at the end of each iteration.
  with metric_writers.ensure_flushes(writer):
    for step in range(initial_step, num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on the GPU/TPU
      # devices.

###################################################################################################################

      "#Schritt 14: Überprüfen, ob ein Szenenwechsel stattfinden soll."
      #print("Step 14")
      if step % step_per_scene == 0:
        scene_idx = np.random.randint(len(scene_path_list))
        logging.info("Loading scene {}".format(scene_path_list[scene_idx]))  # pylint: disable=logging-format-interpolation
        curr_scene = scene_path_list[scene_idx]
        if config.dataset.name == "dtu":
          # lighting can take values between 0 and 6 (both included)
          config.dataset.dtu_light_idx = np.random.randint(low=0, high=7)
        train_ds = datasets.create_train_dataset(config, curr_scene)
        ptrain_ds = flax.jax_utils.prefetch_to_device(train_ds, 6)
##################################################################################################################
      "#Schritt 15: Überprüfen, ob der aktuelle Schritt der letzte Schritt ist."
      is_last_step = step == num_train_steps
##################################################################################################################
      #This allows for profiling and tracing the execution of the training step.
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        "#Schritt 16: Ausführen eines Trainingsschritts."
        #print("Step 16")
        batch = next(ptrain_ds)
        state, metrics_update, keys = p_train_step(
            rng=keys, state=state, batch=batch)
        metric_update = flax_utils.unreplicate(metrics_update)#to ensure consistency across the distributed setup.
        train_metrics = (#The updated metrics are merged into the train_metrics object.
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      ##################################################################################################################
      #Schritt 17: Logging und Speichern von Metriken
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)#nur für die ersten 5 Schritte protokolliert.

      #Die Schleife iteriert über die hooks-Liste, die verschiedene Hooks oder Aktionen enthält, die bei jedem Trainingsschritt ausgeführt werden sollen.
      #Diese Hooks können Aktionen wie Fortschrittsberichte, Aktualisierung der Lernraten oder andere benutzerdefinierte Operationen umfassen.
      for h in hooks:
          h(step)

      if step % config.train.log_loss_every_steps == 0 or is_last_step:
        #Die Methode write_scalars des writer-Objekts wird aufgerufen, um die berechneten Metriken zu speichern.
        #Die Metriken werden mit dem aktuellen Schritt verknüpft.
        writer.write_scalars(step, train_metrics.compute())

        # Berechne die Metriken
        log_dict = metric_update.compute()
        # Wandele die Metriken in ein JSON-freundliches Format um
        for k, v in log_dict.items():
            log_dict[k] = v.item()

        train_loss_list.append(log_dict["train_loss"])
        train_psnr_list.append(log_dict["train_psnr"])
        weight_l2_list.append(log_dict["weight_l2"])
        learining_rate_list.append(log_dict["learining_rate"])
        total_loss_list.append(log_dict["total_loss"])
        train_loss_std_list.append(log_dict["train_loss_std"])
        train_loss_c_list.append(log_dict["train_loss_c"])
        train_psnr_c_list.append(log_dict["train_psnr_c"])

        #Das Objekt train_metrics wird auf None zurückgesetzt, um es auf die nächsten Metriken vorzubereiten.
        train_metrics = None

     ###################################################################################################################
      #print("Step 18")
      #Schritt 18: Ausführen einer Evaluation.
      if step % config.train.render_every_steps == 0: #or is_last_step:
        test_batch = next(eval_ds)
        #Das Bild (test_batch.target_view.rgb) wird in den Fließkommawertebereich umgewandelt, um für die Evaluation verwendet zu werden.
        test_pixels = model_utils.uint2float(
            test_batch.target_view.rgb)  # extract for evaluation
        with report_progress.timed("eval"):
          pred_color, pred_disp, pred_acc = eval_step(state, keys[0],
                                                      test_batch, render_pfn,
                                                      config)
 #################################################################################################################
        #------------------------------------------------------------------
        # Log metrics and images for host 0
        #Die Evaluationsergebnisse werden nur für Host 0 geloggt, um Redundanz zu vermeiden. In einem verteilten Trainingsszenario wird das Training auf mehreren Geräten oder Hosts parallel durchgeführt.
        #Jeder Host berechnet unabhängig voneinander seine eigenen Evaluationsergebnisse. Um eine doppelte Aufzeichnung der Ergebnisse zu verhindern und den Speicherbedarf zu reduzieren, wird festgelegt, dass nur Host 0 die Ergebnisse loggt.
        #Host 0 wird oft als "Master"-Host bezeichnet und ist für die Koordination und das Management der Trainingsschritte verantwortlich. Es ist üblich, dass Host 0 zusätzliche Aufgaben übernimmt,
        #wie z.B. das Speichern von Metriken, das Schreiben von Protokollen oder das Erstellen von visuellen Darstellungen. Die anderen Hosts konzentrieren sich hauptsächlich auf die Berechnung der Schritte und den Austausch von Parametern.
        #Durch die Beschränkung des Loggings auf Host 0 wird sichergestellt, dass die Evaluationsergebnisse nur einmal erfasst und gespeichert werden, was effizienter ist und den Arbeitsaufwand verringert.

        print("Step 19")
        #Schritt 19: Logging der Evaluationsergebnisse für Host 0.
        #------------------------------------------------------------------
        #Die Bedingung jax.process_index() == 0 überprüft, ob der aktuelle Host den Index 0 hat.
        #Dies stellt sicher, dass nur Host 0 die folgenden Schritte ausführt und die Evaluationsergebnisse loggt.
        if jax.process_index() == 0:
          #der quadratische Fehler zwischen den vorhergesagten Farbwerten (pred_color)
          #und den tatsächlichen Farbwerten (test_pixels) gemittelt und anschließend die PSNR berechnet.
          psnr = model_utils.compute_psnr(
              ((pred_color - test_pixels)**2).mean())
          ssim = 0.#Hier könnte eine Berechnung der SSIM (Structural Similarity Index) erfolgen, sofern implementiert.

          # die berechneten Metriken in das Metrik-Logging geschrieben.
          writer.write_scalars(step, {
              "train_eval/test_psnr": psnr,
              "train_eval/test_ssim": ssim,
          })
          #Bilder in das Bild-Logging geschrieben.
          writer.write_images(
              step, {
                  "test_pred_color": pred_color[None, :],
                  "test_target": test_pixels[None, :]
              })
          if pred_disp is not None:
            writer.write_images(step, {"test_pred_disp": pred_disp[None, :]})
          if pred_acc is not None:
            writer.write_images(step, {"test_pred_acc": pred_acc[None, :]})
        #------------------------------------------------------------------

      #Schritt 20: Speichern von Checkpoints.
      #Dieser Schritt wird nur vom Host mit Index 0 durchgeführt.
      if (jax.process_index()
         == 0) and (step % config.train.checkpoint_every_steps == 0 or
                    is_last_step):
       # Write final metrics to file
       #Dazu werden die berechneten Metriken aus dem metric_update-Objekt extrahiert und in ein JSON-Format umgewandelt.
       #Die Metriken werden in einer Datei mit dem Namen "train_logs.json" im Arbeitsverzeichnis gespeichert.

       # Generiere einen eindeutigen Dateinamen für jede Iteration
       filename = f"train_logs_{step}.json"  # Annahme: 'step' ist definiert
       train_loss_file = f"train_loss.json"
       train_psnr_file = f"train_psnr.json"
       weight_l2_file = f"weights_l2.json"
       learning_rate_file = f"lr.json"
       total_loss_file = f"total_loss.json"
       train_loss_std_file = f"train_loss_std.json"
       train_loss_c_file = "train_loss_c.json"
       train_psnr_c_file = "train_psnr_c.json"

       with file_utils.open_file(os.path.join(workdir, train_loss_file), "w") as f:
           f.write(json.dumps(train_loss_list))
       with file_utils.open_file(os.path.join(workdir, train_psnr_file), "w") as f:
           f.write(json.dumps(train_psnr_list))
       with file_utils.open_file(os.path.join(workdir, weight_l2_file), "w") as f:
           f.write(json.dumps(weight_l2_list))
       with file_utils.open_file(os.path.join(workdir, learning_rate_file), "w") as f:
           f.write(json.dumps(learining_rate_list))
       with file_utils.open_file(os.path.join(workdir, total_loss_file), "w") as f:
           f.write(json.dumps(total_loss_list))
       with file_utils.open_file(os.path.join(workdir, train_loss_std_file), "w") as f:
           f.write(json.dumps(train_loss_std_file))
       with file_utils.open_file(os.path.join(workdir, train_loss_c_file), "w") as f:
           f.write(json.dumps(train_loss_c_file))
       with file_utils.open_file(os.path.join(workdir, train_psnr_c_file), "w") as f:
           f.write(json.dumps(train_psnr_c_file))


       #with file_utils.open_file(os.path.join(workdir, "train_logs.json"), "w") as f:
       with file_utils.open_file(os.path.join(workdir, filename), "w") as f:
        log_dict = metric_update.compute()
        for k, v in log_dict.items():
          log_dict[k] = v.item()
        f.write(json.dumps(log_dict))


       with report_progress.timed("checkpoint"):
         state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
         checkpoints.save_checkpoint(workdir, state_to_save, step, keep=50)#Es werden maximal 100 Checkpoints aufbewahrt, um den Speicherplatz zu begrenzen.

  #Schritt 21: Abschluss der Trainingsschleife.
  logging.info("Finishing training at step %d", num_train_steps)
