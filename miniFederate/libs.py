def addModel(destinationModel, sourceModel) -> torch.nn.Module:
    """
    Performs an addition of the parameters of two models
    Args:
        destinationModel (torch.nn.Module)
        sourceModel (torch.nn.Module)
    Returns:
        torch.nn.Module
    """

    params1 = sourceModel.named_parameters()
    params2 = destinationModel.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(
                    param1.data + dict_params2[name1].data)
    return destinationModel


def scaleModel(model, scale) -> torch.nn.Module:
    """
    Scales the parameters of the model.
    Args:
        model (torch.nn.Module)
        scale (float)
    Returns:
        torch.nn.Module:
    """
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model
