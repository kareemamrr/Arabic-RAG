import os
import re

import camelot
import cv2
import pandas as pd
from PIL import Image

# Load PDF
from surya.input.load import load_pdf

# OCR
from surya.ocr import run_ocr
from surya.model.detection import segformer
from surya.model.recognition.model import load_model as load_ocr_model
from surya.model.recognition.processor import load_processor as load_ocr_processor

# Layout analysis
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.segformer import (
    load_model as load_detection_model,
    load_processor as load_detection_processor,
)

# Reading order
from surya.ordering import batch_ordering
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.model.ordering.model import load_model as load_order_model

from surya.settings import settings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PDF_PATH = "data/pdf1/Press Release - 2022 Results (Stock Market).pdf"
ORIGINAL_IMGS_PATH = "data/pdf1/original_images"
CROPPED_IMGS_PATH = "data/pdf1/cropped_images"
TEXT_FILE_PATH = "data/pdf1/pdf1.txt"
START_PAGE = 1  # zero-based
SEGMENT = True


class BasePreprocesser:
    def __init__(
        self,
        pdf_path: str,
        original_imgs_path: str = ORIGINAL_IMGS_PATH,
        cropped_imgs_path: str = CROPPED_IMGS_PATH,
        text_file_path: str = TEXT_FILE_PATH,
        start_page: int = START_PAGE,
        segment: bool = SEGMENT,
    ):

        # add assert for pdf
        self.pdf_path = pdf_path
        self.original_imgs_path = original_imgs_path
        self.cropped_imgs_path = cropped_imgs_path
        self.text_file_path = text_file_path
        self.start_page = start_page
        self.segment = SEGMENT

    def run_pipeline(self):
        self._create_dirs()
        self._pdf_to_png()
        self._layout_analysis()
        self._reading_order()
        self._save_cropped_imgs()
        self._run_ocr()
        self._clean_text_file()

    def _create_dirs(self):
        _ = [
            os.makedirs(path)
            for path in [self.cropped_imgs_path, self.original_imgs_path]
            if not os.path.exists(path)
        ]

    def _get_files_in_dir(self, dir_path: str):
        return sorted([f"{dir_path}/{file_path}" for file_path in os.listdir(dir_path)])

    def _parse_table_locations(self, table_locations: set):
        return ",".join(map(str, list(table_locations)))

    def _pdf_to_png(self):
        imgs, _ = load_pdf(self.pdf_path, start_page=self.start_page)
        _ = [
            img.save(f"{self.original_imgs_path}/{idx}.png")
            for idx, img in enumerate(imgs, self.start_page + 1)
        ]

    def extract_number(self, path: str):
        filename = path.split("/")[-1]
        return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", filename)]

    def _is_number(self, substring: str):
        cleaned_substring = re.sub(r"\s*\.?\s*", "", substring)
        try:
            cleaned_substring = cleaned_substring.strip("()")
            cleaned_substring = cleaned_substring.replace(",", "")
            float(cleaned_substring.replace("%", ""))
            return True
        except ValueError:
            return False

    def _clean_text(self, df: pd.DataFrame, slice=False):
        if not slice:
            df.iloc[:, 3] = df.iloc[:, 3].str.replace("\n", "")
        else:
            df["التفير %"] = df["التفير %"].str[:-2]
        return df

    def _layout_analysis(self):
        paths = self._get_files_in_dir(self.original_imgs_path)
        images = [Image.open(path) for path in paths]
        model = load_detection_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        processor = load_detection_processor(
            checkpoint=settings.LAYOUT_MODEL_CHECKPOINT
        )
        det_model = load_detection_model()
        det_processor = load_detection_processor()

        line_predictions = batch_text_detection(images, det_model, det_processor)
        self.layout_predictions = batch_layout_detection(
            images, model, processor, line_predictions
        )

    def _reading_order(self):
        model = load_order_model()
        processor = load_order_processor()
        self.pages_blocks = []

        for page_num in range(len(self.layout_predictions)):
            image = Image.open(f"{self.original_imgs_path}/{page_num + self.start_page + 1}.png")
            bboxes = [bbox.bbox for bbox in self.layout_predictions[page_num].bboxes]
            self.order_predictions = batch_ordering([image], [bboxes], model, processor)

            ll = [
                (bbox_1.position, bbox_2.label, bbox_2.bbox)
                for bbox_1, bbox_2 in zip(
                    self.order_predictions[0].bboxes,
                    self.layout_predictions[page_num].bboxes,
                )
            ]
            sorted_list = sorted(ll, key=lambda x: x[0])
            self.pages_blocks.append(sorted_list)

    def _save_cropped_imgs(self):
        paths = self._get_files_in_dir(self.original_imgs_path)
        output_dir = self.cropped_imgs_path
        images = [cv2.imread(path) for path in paths]
        table_locations = []

        def crop_and_save(image, bbox, label, page_num, bbox_num):
            x1, y1, x2, y2 = bbox
            if y2 - y1 > 10 and x2 - x1 > 10:
                cropped_image = image[y1:y2, x1:x2]
                if self.segment:
                    filename = f"{page_num}-{bbox_num}_{label}.png"
                else:
                    filename = f"{page_num}-{bbox_num}_Text.png"
                cv2.imwrite(os.path.join(output_dir, filename), cropped_image)

        for page_num, page in enumerate(self.pages_blocks):
            for block_id in page:
                pos = block_id[0]
                label = block_id[1]
                bbox = block_id[2]
                if label in ["Text", "Title", "Section-header"]:
                    crop_and_save(images[page_num], bbox, label, page_num + 2, pos)
                elif label == "Table":
                    table_locations.append(page_num + 2)
                    crop_and_save(images[page_num], bbox, label, page_num + 2, pos)

        self.table_locations = self._parse_table_locations(set(table_locations))

    def _run_ocr(self):
        det_processor, det_model = segformer.load_processor(), segformer.load_model()
        rec_model, rec_processor = load_ocr_model(), load_ocr_processor()

        text_paths, tables_paths = self._retrieve_sections_paths()

        images = [Image.open(path) for path in text_paths]
        langs = [["ar"]] * len(images)
        self.text_predictions = run_ocr(
            images, langs, det_model, det_processor, rec_model, rec_processor
        )
        self._write_to_file(key=1)

        if self.segment:
            images = [Image.open(path) for path in tables_paths]
            langs = [["ar"]] * len(images)
            self.table_predictions = run_ocr(
                images, langs, det_model, det_processor, rec_model, rec_processor
            )
            self._parse_table_content()
            self._write_to_file(key=2)

    def _retrieve_sections_paths(self):
        text_paths = sorted(
            [
                f"{self.cropped_imgs_path}/{path}"
                for path in os.listdir(self.cropped_imgs_path)
                if not path.endswith("_Table.png")
            ],
            key=self.extract_number,
        )

        tables_paths = sorted(
            [
                f"{self.cropped_imgs_path}/{path}"
                for path in os.listdir(self.cropped_imgs_path)
                if path.endswith("_Table.png")
            ],
            key=self.extract_number,
        )

        return text_paths, tables_paths

    def _write_to_file(self, key: int):
        if key == 1:
            with open(self.text_file_path, "w", encoding="utf-8", errors="ignore") as f:
                for page in self.text_predictions:
                    for text_line in page.text_lines:
                        f.write(text_line.text + "\n")
        elif key == 2:
            with open(self.text_file_path, "a", encoding="utf-8", errors="ignore") as f:
                for _, row in self.df_filtered.iterrows():
                    f.write(
                        f"اﻟﺮﺑﺢ و اﻟﺨﺴﺎرة {row['مليون ريال سعودي']}: 'العام المالي 2022 {row['العام المالي 2022']}, العام المالي 2021 {row['العام المالي 2021']}, التفير {row['التفير %']}. \n"
                    )

    def _parse_table_content(self):
        table_page_nums = ",".join(map(str, list(self.table_locations)))
        table_text = ""

        for text_line in self.table_predictions[0].text_lines:
            table_text += f"{text_line.text}\n"

        tables = camelot.read_pdf(self.pdf_path, flavor="stream", pages=table_page_nums)
        df = tables[0].df

        p = "\n".join(table_text.split("\n")[8:])

        column_vals = []
        lines = p.split("\n")
        for a in range(len(lines)):
            if not self._is_number(lines[a]) and len(lines[a]) > 1:
                column_vals.append(lines[a])

        rows_offset = df.shape[0] - len(column_vals)

        df.loc[rows_offset:, 4] = column_vals
        df[3] = df[3] + df[4]
        df = df.drop(columns=4)
        df.columns = [
            "التفير %",
            "العام المالي 2021",
            "العام المالي 2022",
            "مليون ريال سعودي",
        ]
        df_filtered = df.loc[3:]

        df_filtered = self._clean_text(df_filtered.copy(), slice=True)
        self.df_filtered = self._clean_text(df_filtered.copy())

    def _clean_text_file(self):
        with open(self.text_file_path, "r+") as file:
            lines = [line for line in file.readlines() if line.strip()]
            file.seek(0)
            file.writelines(lines)
            file.truncate()


if __name__ == "__main__":
    preprocessor = BasePreprocesser(
        pdf_path=PDF_PATH,
        original_imgs_path=ORIGINAL_IMGS_PATH,
        cropped_imgs_path=CROPPED_IMGS_PATH,
        text_file_path=TEXT_FILE_PATH,
        start_page=START_PAGE,
        segment=SEGMENT,
    )
    preprocessor.run_pipeline()
