import numpy as np
import torch


def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_huber_loss(preds, labels, null_val=np.nan, delta=1.0):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)  # torch.le: less than or equal to
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def MSE(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~np.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def RMSE(pred, true, null_value=np.nan):
    return torch.sqrt(MSE(pred, true, null_value))


def MAPE(preds, labels, null_val: float = np.nan):
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = torch.where(torch.abs(labels) < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.full_like(labels, fill_value=null_val), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)  # ne: not equal缩写
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)  # ge: greater than or equal的缩写
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val))

def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        mask &= labels.ge(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def MSPE(preds, labels, null_val: float = np.nan):
    """Masked mean square percentage error.

        Args:
            preds (torch.Tensor): predicted values
            labels (torch.Tensor): labels
            null_val (float, optional): null value.
                                        In the mape metric, null_val is set to 0.0 by all default.
                                        We keep this parameter for consistency, but we do not allow it to be changed.
                                        Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

        Returns:
            torch.Tensor: masked mean absolute percentage error
        """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    labels = torch.where(torch.abs(labels) < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.full_like(labels, fill_value=null_val), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, true, null_value, mask_value):
    # mae = masked_mae(pred, true, 0.0)
    mae = masked_mae_torch(pred, true, null_value, mask_value)
    # mse = MSE(pred, true, 0.0)
    # mse = masked_mse_torch(pred, true, null_value, mask_value)
    # rmse = RMSE(pred, true, 0.0)
    rmse = masked_rmse_torch(pred, true, null_value, mask_value)
    # mape = MAPE(pred, true)
    mape = masked_mape_torch(pred, true, null_value, mask_value)

    return mae, rmse, mape
