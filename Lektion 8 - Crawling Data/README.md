# Possible error
If you run into the following error you’ll want to navigate to the core.py file specified and change line 690 to: photoset = list(rsp)[0]

File "C:\python\lib\site-packages\flickrapi\core.py", line 690, in data_walker
photoset = rsp.getchildren()[0]
AttributeError: 'xml.etree.ElementTree.Element' object has no attribute 'getchildren'

# pyimgdata

These are the Flickr License identifiers for images.

* 0 All Rights Reserved
* 1 Attribution-NonCommercial-ShareAlike License
* 2 Attribution-NonCommercial License
* 3 Attribution-NonCommercial-NoDerivs License
* 4 Attribution License
* 5 Attribution-ShareAlike License
* 6 Attribution-NoDerivs License
* 7 No known copyright restrictions
* 8 United States Government Work
* 9 Public Domain Dedication (CC0)
* 10 Public Domain Mark
