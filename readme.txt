1. Data Format of the Annotation
�EFile Name: ***.json (*** = same as the corresponding image file name(*** in ***.tif).)
�EDescription:
  - coastline_points: [[x1, y1], ...]
  - validate_lines: [[[x1, y1], [x2, y2]], ...]
�ENotes:
  - The origin of the coordinate system in the image is top left.
  - In "coastline_points", [x1, y1] corresponds to [x-coordinate(horizontal), y-coordinate(vertical)] in the image.
  - For each image, the coastline is expressed as the set of the coordinates as in the description.
  - In "validate_lines", [x1, y1] and [x2, y2] correspond to the end points of the validate line(line segment).
  - Please use "validate_lines" for validating your algorithm.


2. Submission File Format
�EFile Name: ***.json (*** = whatever name you like(e.g. submit))
�EDescription:
  - image_file_0:[[x1, y1], ...]
  - image_file_1:[[x1, y1], ...]
  ...
�ENotes:
  - The origin of the coordinate system in the image is top left.
  - [x1, y1] corresponds to [x-coordinate(horizontal), y-coordinate(vertical)] in the image.
  - For each image, the predicted coastline is expressed as the set of the coordinates as in the description.
  - For each image, the number of predicted points must not exceed 30000.
  - Please also refer to "sample_submit.json".