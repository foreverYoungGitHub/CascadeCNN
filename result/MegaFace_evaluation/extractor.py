import json

from pprint import pprint

#with open('/home/xileli/Documents/dateset/megaface/data/daniel/FlickrFinal2/079/079A02.JPG.json') as data_file:
with open('/home/xileli/Documents/dateset/megaface/data/daniel/FlickrFinal2/103/10300581@N07/5504068610_0.jpg.json') as data_file:
    data = json.load(data_file)

pprint(data)

if 'bounding_box' in data:
    pprint(data['bounding_box'])

