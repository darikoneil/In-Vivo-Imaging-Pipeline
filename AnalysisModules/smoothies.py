
import numpy
from scipy.ndimage.filters import gaussian_filter




def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):


    if option == 1:
        def condgradient(delta, spacing):
            return numpy.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return numpy.where(numpy.abs(delta) <= kappa_s, top, 0)


    out = numpy.array(img, dtype=numpy.float32, copy=True)


    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)


    deltas = [numpy.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):


        for i in range(out.ndim):
            slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            deltas[i][tuple(slicer)] = numpy.diff(out, axis=i)


        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]


        for i in range(out.ndim):
            slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            matrices[i][tuple(slicer)] = numpy.diff(matrices[i], axis=i)


        out += gamma * (numpy.sum(matrices, axis=0))

    return out