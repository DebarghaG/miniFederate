import torch
import torch.nn as nn
import libs import addModel
import scaleModel


def federated_avg(models: Dict[Any, torch.nn.Module]) -> torch.nn.Module:
    """
    Performs a federated averaging step on a dictionary containing models.

    Args:
        models (Dict[Any, torch.nn.Module])
    Returns:
        torch.nn.Module
    """
    nr_models = len(models)
    model_list = list(models.values())
    model = type(model_list[0])()
    for i in range(nr_models):
        model = addModel(model, model_list[i])
    model = scaleModel(model, 1.0 / nr_models)
    return model
