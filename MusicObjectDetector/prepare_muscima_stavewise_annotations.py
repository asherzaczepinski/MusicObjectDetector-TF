import json
import os
import shutil
from glob import glob
from typing import Tuple, List, Dict

import pandas
from PIL import Image
from mung.io import read_nodes_from_file
from mung.node import Node
from tqdm import tqdm

from muscima_annotation_generator import create_annotations_in_pascal_voc_format


def cut_images(muscima_pp_dataset_directory: str, output_path: str,
               exported_annotations_file_path: str, annotations_path: str):
    muscima_image_directory = os.path.join(muscima_pp_dataset_directory, "v2.0", "data", "images", "*.png")
    os.makedirs(output_path, exist_ok=True)
    if os.path.exists(exported_annotations_file_path):
        os.remove(exported_annotations_file_path)
    shutil.rmtree(annotations_path, ignore_errors=True)

    annotations_dictionary = load_all_muscima_annotations(muscima_pp_dataset_directory)
    crop_annotations = []
    croppings = []

    image_paths = glob(muscima_image_directory)
    for image_path in tqdm(image_paths, desc="Cutting images"):
        image_name = os.path.basename(image_path)[:-4]  # cut away the extension .png
        image = Image.open(image_path, "r")  # type: Image.Image
        image_width = image.width
        image_height = image.height
        objects_appearing_in_image: List[Node] = None

        for document, crop_objects in annotations_dictionary.items():
            if image_name in document:
                objects_appearing_in_image = crop_objects
                break

        if objects_appearing_in_image is None:
            # Image has annotated staff-lines, but does not have corresponding crop-object annotations, so skip it
            continue

        staff_objects = [x for x in objects_appearing_in_image if x.class_name == "staff"]
        max_offset_before_first_and_after_last_staff = 120

        if staff_objects is None:
            # Image has no staff lines -> Report error
            print("Error: Image {0} has no annotated staff lines".format(image_path))
            continue

        next_y_top = max(0, staff_objects[0].top - max_offset_before_first_and_after_last_staff)
        last_bottom = min(staff_objects[len(staff_objects) - 1].bottom + max_offset_before_first_and_after_last_staff,
                          image_height)

        output_image_counter = 1
        crop_positions = []
        for staff_index in range(len(staff_objects)):
            staff = staff_objects[staff_index]
            if staff_index < len(staff_objects) - 1:
                y_bottom = staff_objects[staff_index + 1].top
            else:
                y_bottom = last_bottom
            top_offset = next_y_top
            next_y_top = staff.bottom
            left_offset = 0

            image_crop_bounding_box_left_top_bottom_right = (left_offset, top_offset, image_width, y_bottom)
            image_crop_bounding_box_top_left_bottom_right = (top_offset, left_offset, y_bottom, image_width)

            file_name = "{0}_{1}.png".format(image_name, output_image_counter)
            output_image_counter += 1

            objects_appearing_in_cropped_image = \
                compute_objects_appearing_in_cropped_image(file_name,
                                                           image_crop_bounding_box_top_left_bottom_right,
                                                           objects_appearing_in_image)

            cropped_image = image.crop(image_crop_bounding_box_left_top_bottom_right).convert('RGB')

            for object_appearing_in_cropped_image in objects_appearing_in_cropped_image:
                file_name = object_appearing_in_cropped_image[0]
                class_name = object_appearing_in_cropped_image[1]
                translated_bounding_box = object_appearing_in_cropped_image[2]
                trans_top, trans_left, trans_bottom, trans_right = translated_bounding_box
                crop_annotations.append([file_name, trans_left, trans_top, trans_right, trans_bottom, class_name])

            create_annotations_in_pascal_voc_format(annotations_path, file_name, objects_appearing_in_cropped_image,
                                                    cropped_image.width, cropped_image.height, 3)

            output_file = os.path.join(output_path, file_name)
            cropped_image.save(output_file, "png")
            crop_positions.append({
                "file_name": file_name,
                "width": cropped_image.width,
                "height": cropped_image.height,
                "left_offset": left_offset,
                "top_offset": top_offset
            })

        croppings.append({
            "image_path": image_path,
            "crops": crop_positions
        })

    with open(os.path.join(output_path, "../croppings.json"), "w") as file:
        json.dump(croppings, file)
    annotation_data = pandas.DataFrame(crop_annotations,
                                       columns=['filename', 'left', 'top', 'right', 'bottom', 'class'])
    annotation_data.to_csv(exported_annotations_file_path, index=False)


def load_all_muscima_annotations(muscima_pp_dataset_directory) -> Dict[str, List[Node]]:
    """
    :param muscima_pp_dataset_directory:
    :return: Returns a dictionary of annotations with the filename as key
    """
    raw_data_directory = os.path.join(muscima_pp_dataset_directory, "v2.0", "data",
                                      "annotations")
    all_xml_files = [y for x in os.walk(raw_data_directory) for y in glob(os.path.join(x[0], '*.xml'))]

    crop_object_annotations = {}
    for xml_file in tqdm(all_xml_files, desc='Parsing annotation files'):
        nodes = read_nodes_from_file(xml_file)
        doc = nodes[0].document
        crop_object_annotations[doc] = nodes
    return crop_object_annotations


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def area(a):
    top, left, bottom, right = a
    return (bottom - top) * (right - left)


def compute_objects_appearing_in_cropped_image(file_name: str,
                                               image_crop_bounding_box_top_left_bottom_right: Tuple[int, int, int, int],
                                               all_music_objects_appearing_in_image: List[Node],
                                               intersection_over_area_threshold_for_inclusion=0.8) \
        -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
    x_translation_for_cropped_image = image_crop_bounding_box_top_left_bottom_right[1]
    y_translation_for_cropped_image = image_crop_bounding_box_top_left_bottom_right[0]

    objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
    for music_object in all_music_objects_appearing_in_image:
        if music_object.class_name in ["staff", "staff_line", "staff_space"]:
            continue

        intersection_over_area = intersection(image_crop_bounding_box_top_left_bottom_right,
                                              music_object.bounding_box) / area(music_object.bounding_box)
        if intersection_over_area > intersection_over_area_threshold_for_inclusion:
            top, left, bottom, right = music_object.bounding_box
            img_top, img_left, img_bottom, img_right = image_crop_bounding_box_top_left_bottom_right
            img_width = img_right - img_left - 1
            img_height = img_bottom - img_top - 1
            translated_bounding_box = (
                max(0, top - y_translation_for_cropped_image),
                max(0, left - x_translation_for_cropped_image),
                min(img_height, bottom - y_translation_for_cropped_image),
                min(img_width, right - x_translation_for_cropped_image))
            objects_appearing_in_cropped_image.append((file_name, music_object.class_name, translated_bounding_box))

    return objects_appearing_in_cropped_image


def compute_statistics(annotations_csv):
    # Create statistics for how many instances of each class exist
    annotations = pandas.read_csv(annotations_csv)
    classes = annotations[['class']].groupby('class').size().reset_index(name='counts')  # type: pandas.DataFrame
    classes.to_csv("data/Class-Statistics.csv", header=True, index=False)


if __name__ == "__main__":
    muscima_pp_dataset_directory = os.path.join("data", "muscima_pp")

    annotations_csv = "data/Stavewise_Annotations.csv"
    cut_images(muscima_pp_dataset_directory,
               "data/muscima_pp_cropped_images_with_stafflines",
               annotations_csv,
               "data/Stavewise_Annotations")

    compute_statistics(annotations_csv)
