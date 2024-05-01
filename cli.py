import argparse
from processor import BasePreprocesser


def main():
    parser = argparse.ArgumentParser(description="PDF to Image Converter")

    parser.add_argument(
        "--pdf_path", type=str, default="input.pdf", help="Path to the input PDF file"
    )
    parser.add_argument(
        "--originals_path",
        type=str,
        default="original_images",
        help="Path to store original images",
    )
    parser.add_argument(
        "--cropped_path",
        type=str,
        default="cropped_images",
        help="Path to store cropped images",
    )
    parser.add_argument(
        "--text_file_path",
        type=str,
        default="output.txt",
        help="Path to store extracted text",
    )
    parser.add_argument(
        "--start_page",
        type=int,
        default=1,
        help="Start page number of PDF for conversion",
    )
    parser.add_argument(
        "--segment", action="store_true", default=False, help="Segment text into blocks"
    )

    args = parser.parse_args()

    processor = BasePreprocesser(
        pdf_path=args.pdf_path,
        original_imgs_path=args.originals_path,
        cropped_imgs_path=args.cropped_path,
        text_file_path=args.text_file_path,
        start_page=args.start_page,
        segment=args.segment,
    )

    processor.run_pipeline()


if __name__ == "__main__":
    main()
