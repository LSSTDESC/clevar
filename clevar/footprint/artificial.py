#!/usr/bin/env python
import numpy as np
import os, sys
from ..utils import hp
from .footprint import Footprint


def create_footprint(ra, dec, nside=None, min_density=2, neighbor_fill=None, nest=False):
    """
    Create footprint from (Ra, Dec). Can compute optimal NSIDE given a density
    and also fill holes.

    Parameters
    ----------
    ra: numpy array
        Ra array in degrees
    dec: numpy array
        Dec array in degrees
    nside: int, None
        Number for healpix NSIDE of pixel
    min_density: float
        Threshold density of obj./pixel to compute nside
        when nside is not defined
    neighbor_fill: int, None
        Minimum number of neighbors to fill a pixel: 1<n<8, optimal is 5.
        If None, the holes are not filled.
    nest: bool
        Nested ordering. If false use ring.

    Returns
    -------
    ftpt: FootprintZmax object
        Footprint
    """
    # actually creates a mask with pixel density < dens
    nside, pixel = (
        nside_from_density(ra, dec, min_density, nest=nest)
        if nside is None
        else (nside, np.array(list(set(hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)))))
    )
    print(f"Footprint NSIDE: {nside:,}")
    print(f"Pixels in footprint: {pixel.size:,}")
    ftpt = Footprint(nside=nside, pixel=pixel, nest=nest)
    # filling holes
    ftpt = ftpt if neighbor_fill is None else fill_holes_conv(ftpt, neighbor_fill, nest=nest)
    print(f'Pixels in footprint: {ftpt["pixel"].size:,}')
    return ftpt


def nside_from_density(ra, dec, min_density, nest=False):
    """
    Compute NSIDE based on a minimum density

    Parameters
    ----------
    ra: numpy array
        Ra array in degrees
    dec: numpy array
        Dec array in degrees
    min_density: float
        Threshold density of obj./pixel
    nest: bool
        Nested ordering. If false use ring.

    Returns
    -------
    nside: int
        Number for healpix NSIDE
    pixel_set: array
        List of pixels in footprint
    """
    nside = 2
    pixel = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    pixel_set = np.array(list(set(pixel)))
    for n in range(2, 12):
        print(f"NSIDE({nside}) -> {pixel.size/pixel_set.size} clusters per pixel")
        if pixel.size / pixel_set.size < min_density:
            return nside, pixel_set
        nside = 2**n
        pixel = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
        pixel_set = np.array(list(set(pixel)))
    raise ValueError(f"NSIDE required > {2**n}")


def fill_holes(ftpt, neighbor_fill, nest=False):
    """
    Fill holes in a footprint mask

    Parameters
    ----------
    ftpt: clevar.mask.Footprint object
        Footprint
    pixels: numpy array
        Array with indices of healpy pixels of the footprint
    neighbor_fill: int
        Minimum number of neighbors to fill a pixel: 1<n<8, optimal is 5
    nest: bool
        Nested ordering. If false use ring.

    Returns
    -------
    ftpt: clevar.mask.Footprint object
        Footprint with holes filled
    """
    all_neighbors = hp.neighbors_of_pixels(ftpt.nside, ftpt["pixel"], nest=nest)
    out_neighbors = all_neighbors[ftpt.get_values_in_pixels("detfrac", all_neighbors, 0) == 0]
    add_pixels = [
        p
        for p in out_neighbors
        if ftpt.get_values_in_pixels(
            "detfrac", hp.neighbors_of_pixels(ftpt.nside, p, nest=nest), 0
        ).sum()
        >= neighbor_fill
    ]
    return (
        ftpt
        if len(add_pixels) == 0
        else Footprint(nside=ftpt.nside, pixel=np.append(ftpt["pixel"], add_pixels), nest=nest)
    )


def fill_holes_conv(ftpt, neighbor_fill, nest=False):
    """
    Interactively fill holes in a footprint mask until convergence, updates ftpt input

    Parameters
    ----------
    ftpt: FootprintZmax object
        Footprint
    neighbor_fill: int
        Minimum number of neighbors to fill a pixel: 1<n<8, optimal is 5.
    nest: bool
        Nested ordering. If false use ring.

    Returns
    -------
    ftpt: clevar.mask.Footprint object
        Footprint with holes filled
    """
    if neighbor_fill is not None:
        len_0 = len_l = ftpt["pixel"].size
        print("### filling holes ###")
        while True:
            print(" - filling")
            ftpt = fill_holes(ftpt, neighbor_fill, nest=nest)
            len_t = ftpt["pixel"].size
            print(f"   size: {len_l:,} -> {len_t:,} (+{len_t-len_l:,})")
            if len_l == len_t:
                break
            len_l = len_t
        print(" Total Change:")
        print(f"   size: {len_0:,} -> {len_t:,} (+{len_t-len_0:,})")
        print("### filled! ###")
    return ftpt
