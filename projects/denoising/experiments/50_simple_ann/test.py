#!/usr/bin/env python
import pickle
from skimage import io
from projects.denoising.experiments.results import _ResultSet

res = _ResultSet('1.json')
try:
    res.results.run_time
except:
    pass

source_image = pickle.loads(res.parameters.source_image_dump)
target_image = pickle.loads(res.parameters.target_image_dump)
filtered_image = pickle.loads(res.results.filtered_image)

io.imshow(source_image)
io.show()

io.imshow(target_image)
io.show()

io.imshow(filtered_image)
io.show()
