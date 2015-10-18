#!/usr/bin/env python2.7

import json
import sys
import argparse
import numpy as np
import cv2

def draw_judgement(im, judgements, delta=1.0):
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = im.shape[0:2]

    for c in comparisons:
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        x1 = int(point1['x'] * cols)
        y1 = int(point1['y'] * rows)
        x2 = int(point2['x'] * cols)
        y2 = int(point2['y'] * rows)
        if darker == '1':
            cv2.arrowedLine(im, (x2, y2), (x1, y1), (0, 0, 255), 2)
        elif darker == '2':
            cv2.arrowedLine(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            'Draw out human judgement on the input image'
        )
    )
    
    parser.add_argument(
        'original', metavar='<original.png>',
        help='original image'
    )

    parser.add_argument(
        'judgements', metavar='<judgements.json>',
        help='human judgements JSON file'
    )

    parser.add_argument(
        'output', metavar='<output.png>',
        help='output filename'
    )

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    im = cv2.imread(args.original)
    judgements = json.load(open(args.judgements))

    gt = draw_judgement(im, judgements)
    cv2.imwrite(args.output, im)
