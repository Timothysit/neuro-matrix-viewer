import xarray as xr
import glob
import os


def get_ds_var_names(ds):
    """
    Get the list of variable names associated with an xarray dataset.
    :param ds:
    :return:
    """
    # From here:
    # https://stackoverflow.com/questions/42805090/loop-through-dataarray-attributes-in-an-xarray-dataset
    var_names = list()
    for varname, da in ds.data_vars.items():
        var_names.append(varname)

    return var_names


def get_ds_dim_names(ds):
    """
    Get the list of dimension names associated with an xarray dataset or dataarray.
    :param ds:
    :return:
    """

    if type(ds) is xr.core.dataarray.DataArray:
        dim_names = list(ds.dims)
    else:
        dim_names = list(ds.dims.keys())

    return dim_names


def load_list_of_xr(folder_path, search_str='plane*.nc'):
    """
    Loads a list of xarray objects.
    :param folder_path:
    :param search_str:
    :return:
    """

    xr_path_list = glob.glob(os.path.join(folder_path, search_str))
    list_of_ds = [xr.open_dataset(fname) for fname in xr_path_list]


    return list_of_ds