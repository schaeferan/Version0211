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

"""Main file for running the example."""

#The app module in Abseil provides functionality for building command-line applications.
#It includes features such as argument parsing, error handling, and application lifecycle management.
#you can define your application's entry point using the absl.app.run() function.
#This function takes your main function as an argument and handles command-line argument parsing and application execution.
from absl import app

#Das flags-Modul von Abseil bietet eine einfache Möglichkeit, Befehlszeilenargumente zu definieren und zu verarbeiten.
#Es erlaubt das Deklarieren von Flags, die Werte enthalten und über die Befehlszeile gesetzt werden können.
#Diese Flags können dann in Ihrem Programm verwendet werden.
from absl import flags

#Mit dem logging-Modul von Abseil können Sie Log-Nachrichten mit unterschiedlichen Log-Leveln (z. B. Fehler, Warnungen, Informationen) erstellen
#und diese Nachrichten formatieren und ausgeben.
from absl import logging

# Required import to setup work units when running through XManager.
# Das platform-Modul in CLU bietet plattformspezifische Informationen
# und Funktionen zur Abfrage von Informationen über das Betriebssystem und die laufende Python-Implementierung.
from clu import platform

import jax

#Das Modul config_flags in ml_collections ermöglicht es, Befehlszeilenflags zu definieren und zu verwalten, die konfigurierbaren Parametern entsprechen.
#Es erlaubt Ihnen, Standardwerte für die Parameter anzugeben und diese mithilfe von Befehlszeilenargumenten zu überschreiben.
from ml_collections import config_flags

import tensorflow as tf
from gen_patch_neural_rendering import eval_lib
from gen_patch_neural_rendering import train_lib

"FLAGS########################################################################################################"
FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "ml_config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_bool("is_train", None, "If true, run train else eval")
flags.mark_flags_as_required(["ml_config", "workdir", "is_train"])
# Flags --jax_backend_target and --jax_xla_backend are available through JAX.
"#############################################################################################################"

def main(argv):

  """Main function for running training or evaluation."""

  #Die Variable argv ist normalerweise eine Liste von Befehlszeilenargumenten, die an ein Python-Skript übergeben werden.
  #Sie enthält die Argumente, die beim Aufruf des Skripts angegeben wurden.
  del argv

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  if FLAGS.jax_backend_target:

    #Anweisung zur Protokollierung von Informationen mithilfe des logging-Moduls.
    # %s ist ein Platzhalter, wird durch FLAGS.jax_backend_target ersetzt
    logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)

    jax_xla_backend = ("None" if FLAGS.jax_xla_backend is None else
                       FLAGS.jax_xla_backend)
    logging.info("Using JAX XLA backend %s", jax_xla_backend)

  #Die Funktion jax.process_index() gibt den Index des aktuellen JAX-Prozesses zurück.
  #gibt die Anzahl der insgesamt verwendeten JAX-Prozesse zurück. Dies ist die Anzahl der parallel arbeitenden Prozesse
  #Die info()-Methode des logging-Moduls erstellt eine Protokollmeldung
  #Informationen über den aktuellen JAX-Prozess und die Gesamtanzahl der verwendeten JAX-Prozesse
  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())

  #FLAGS.jax_backend_target ist eine Variable, die den Wert des Kommandozeilenarguments --jax_backend_target speichert.
  if FLAGS.is_train:
    # Add a note so that we can tell which Borg task is which JAX host.
    # (Borg task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()},"
        f"process_count: {jax.process_count()}")

    #Zusammengefasst erstellt diese Codezeile ein Artefakt vom Typ Verzeichnis in der Arbeitsumgebung (work_unit)
    #mit dem angegebenen Verzeichnispfad (FLAGS.workdir) und einem optionalen Bezeichner ("workdir").
    #erstellt ein Artefakt mit dem Typ "DIRECTORY" (Verzeichnis) in einem Arbeitsobjekt (work_unit) des Moduls platform.
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         FLAGS.workdir, "workdir")

    #Das erste Argument FLAGS.ml_config ist vermutlich eine Konfigurationsdatei oder ein Konfigurationsobjekt,
    #das die spezifischen Parameter und Einstellungen für das Training und die Auswertung des Modells enthält.
    #Das zweite Argument FLAGS.workdir ist vermutlich der Pfad oder die Adresse des Arbeitsverzeichnisses,
    #in dem das Training und die Auswertung stattfinden sollen.
    #Die Funktion train_and_evaluate wird aufgerufen, um das Training und die Auswertung des Modells durchzuführen,
    #basierend auf den angegebenen Konfigurationen und dem Arbeitsverzeichnis.
    train_lib.train_and_evaluate(FLAGS.ml_config, FLAGS.workdir)

  else:
    eval_lib.evaluate(FLAGS.ml_config, FLAGS.workdir)


if __name__ == "__main__":
  # Provide access to --jax_backend_target and --jax_xla_backend flags.
  jax.config.config_with_absl()
  app.run(main)
