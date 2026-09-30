"""
picasso.render.gpu
~~~~~~~~~~~~~~~~~~

GPU splat backends. Importing this package requires ``wgpu``; the
selection logic in ``picasso.render.backend`` guards the import and
falls back to the CPU backend when it fails.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from .backend_wgpu import WgpuBackend
