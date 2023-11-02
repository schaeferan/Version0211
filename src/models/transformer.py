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

"""
Transformer model.
"""

#"Schritt 1: Importe und Typdefinitionen"

from typing import Any, Callable, Optional, Tuple

from flax import linen as nn

from gen_patch_neural_rendering.src.utils import config_utils

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

##################################################################################################
#"Schritt 2: Hilfsfunktion _resolve"

def _resolve(a, b):
  """
  Returns a if a is not None, else returns b.
  """
  if a is not None:
    return a
  else:
    return b


# The function is tweaked from
# https://github.com/google/flax/blob/main/examples/wmt/models.py
###################################################################################################
#"Schritt 3: Klasse Mlp (MLP-Block)"

class Mlp(nn.Module):
  """
  Transformer MLP block with single hidden layer.
  """

  hidden_params: Optional[int] = None
  out_params: Optional[int] = None
  dropout_rate: float = 0.
  #kernel=gewichte
  #Die Variable kernel_init ist vom Typ Callable[[PRNGKey, Shape, Dtype], Array], was bedeutet, dass sie eine Funktion erwartet,
  #die drei Argumente erhält (PRNGKey, Shape, Dtype) und ein Array (Array) zurückgibt.
  #nn.initializers.xavier_uniform() als Standardwert für kernel_init verwendet.
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.xavier_uniform())

  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      nn.initializers.normal(stddev=1e-6))

  @nn.compact
  #Die @nn.compact-Annotation wird in Flax verwendet, um kompakte Modelle zu definieren. Ein kompaktes Modell ist eine Klasse,
  #die das nn.Module-Interface implementiert und spezielle Methoden verwendet, um die Netzwerkarchitektur zu definieren.
  #Es bietet eine kompakte und intuitive Möglichkeit, Modelle zu definieren, indem die Netzwerkstruktur innerhalb der Klasse selbst deklariert wird.

  #Die Verwendung der Methode __call__ ermöglicht eine elegante und intuitive Nutzung von Objekten einer Klasse,
  #als ob sie eine Funktion wären.
  def __call__(self, inputs, deterministic):
    """
    Wendet den MLP-Block auf die Eingabe an.

    Args:
        inputs: Eingabedaten.
        deterministic: Gibt an, ob der Vorgang deterministisch sein soll.

    Returns:
        Wert: Ausgabedaten.
    """
    h = nn.Dense(
        features=_resolve(self.hidden_params, inputs.shape[-1]),#resolve aus Schritt 2
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    h = nn.gelu(h)# "Gaussian Error Linear Unit"
    h = nn.Dense(
        features=_resolve(self.out_params, inputs.shape[-1]),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            h)
    return h

#####################################################################################################
#"Schritt 4: Klasse SelfAttentionTransformerLayer (Transformer-Schicht)"


class SelfAttentionTransformerLayer(nn.Module):
  """Transformer layer."""

  #attention heads allow the model to focus on different parts of the input sequence simultaneously.
  #By using multiple attention heads, the model can capture different types of relationships and improve its ability to model complex patterns in the input sequence.
  # It allows the model to attend to different aspects of the data in parallel,
  attention_heads: int
  qkv_params: Optional[int] = None
  mlp_params: Optional[int] = None
  dropout_rate: float = 0.


  #Zusammenfassend wendet die __call__-Methode von SelfAttentionTransformerLayer die Self-Attention auf die Eingabe query an,
  #normalisiert die Aufmerksamkeitsausgabe, führt sie durch ein MLP und gibt die normierte Summe der Aufmerksamkeitsausgabe
  #und MLP-Ausgabe als endgültige Ausgabe der Schicht zurück.

  #@nn.compact: Dies ist ein Dekorator, der angibt, dass die folgende Methode als kompaktes Flax-Modul implementiert ist.
  #Ein kompaktes Modul ist eine kompakte Darstellung eines Moduls, bei dem die Parameter automatisch erstellt und initialisiert werden
  @nn.compact
  def __call__(self, query, deterministic):
    """
    Wendet die Transformer-Schicht auf die Eingabe 'query' an.

    Args:
        query: Eingabe für die Self-Attention-Schicht.
        deterministic: Gibt an, ob der Vorgang deterministisch sein soll.

    Returns:
        Wert: Ausgabedaten.
    """

    #query: input to the self-attention layer
    out_params = query.shape[-1]

    # Attention from query to value
    attention_output = nn.SelfAttention(
        num_heads=self.attention_heads,
        qkv_features=self.qkv_params,#qkv_features repräsentiert die Dimensionalität der Query-, Key- und Value-Projektionen
        out_features=out_params,
        dropout_rate=self.dropout_rate)(
            query, deterministic=deterministic)
    normalized_attention_output = nn.LayerNorm()(query + attention_output)

    mlp_output = Mlp(
        hidden_params=self.mlp_params,
        out_params=out_params,
        dropout_rate=self.dropout_rate)(
            normalized_attention_output, deterministic=deterministic)
    return nn.LayerNorm()(normalized_attention_output + mlp_output)

####################################################################################################
#"Schritt 5: Klasse "SelfAttentionTransformer" (Transformer-Modell)"

class SelfAttentionTransformer(nn.Module):
  """Self Attention Transformer."""
  params: config_utils.TransformerParams  # Network parameters.

#Zusammenfassend führt die __call__-Methode des SelfAttentionTransformer-Modells eine Schleife über die Anzahl der Schichten aus
#und wendet in jedem Schleifendurchlauf eine Instanz der SelfAttentionTransformerLayer auf die Eingabe an.
#Das Ergebnis nach Durchlaufen aller Schichten wird als Ausgabe des Modells zurückgegeben.
  @nn.compact
  def __call__(self, points, deterministic):
    """
    Wendet den Transformer auf eine Menge von Eingaben 'points' an.

    Args:
        points: Eingabedaten.
        deterministic: Gibt an, ob der Vorgang deterministisch sein soll.

    Returns:
        Wert: Ausgabedaten.
    """
    for _ in range(self.params.num_layers):
      points = SelfAttentionTransformerLayer(
          attention_heads=self.params.attention_heads,
          qkv_params=self.params.qkv_params,
          mlp_params=self.params.mlp_params,
          dropout_rate=self.params.dropout_rate)(
              query=points, deterministic=deterministic)
    return points
