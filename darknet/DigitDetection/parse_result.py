import json
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True,
                    help='path to yolov4 result file')
parser.add_argument('--output', type=str, required=True,
                    help='output file name')

args = parser.parse_args()
in_file = args.input
out_file = args.output


def sorted_by_key(dict_results):
    keys = dict_results.keys()
    keys = sorted(keys)
    return [dict_results[key] for key in keys]


with open(in_file, 'r') as f:
    results = json.load(f)

    # {'img_name1':{$bbox}, 'img_name2':{$bbox}}
    aggregate_results = {}
    for result in results:
        # which image
        name = result['filename'].split('/')[-1].split('.')[0]
        if name.isdigit():
            name = int(name)

        # create dictionary for this image: bbox coordinates, bbox confidence, bbox class
        aggregate_results[name] = {'bbox': [], 'score': [], 'label': []}

        # read the image size to scale to proper coordinates
        img = cv2.imread(result['filename'])
        img_h, img_w, _ = img.shape

        # process all detected bounding boxes
        for bbox in result['objects']:
            aggregate_results[name]['label'].append(int(bbox['name']))
            aggregate_results[name]['score'].append(bbox['confidence'])

            # turn relative center x, center y, height and width to absolute coordinates (y1, x1, y2, x2)
            y1 = round(
                img_h * (bbox['relative_coordinates']['center_y'] - bbox['relative_coordinates']['height'] * 0.5)
            )
            x1 = round(
                img_w * (bbox['relative_coordinates']['center_x'] - bbox['relative_coordinates']['width'] * 0.5)
            )
            y2 = round(
                img_h * (bbox['relative_coordinates']['center_y'] + bbox['relative_coordinates']['height'] * 0.5)
            )
            x2 = round(
                img_w * (bbox['relative_coordinates']['center_x'] + bbox['relative_coordinates']['width'] * 0.5)
            )
            aggregate_results[name]['bbox'].append([y1, x1, y2, x2])

    # aggregated results are written to output json files
    with open(out_file, 'w') as f_out:
        json.dump(sorted_by_key(aggregate_results), f_out)
        print(sorted_by_key(aggregate_results))

    print('done!')
