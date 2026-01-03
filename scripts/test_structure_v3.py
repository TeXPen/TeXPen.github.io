import os
import sys
import cv2
import numpy as np
import json
import types
from unittest.mock import MagicMock


# Mock torch only to avoid DLL load errors from broken torch installation
# PaddleOCR inference usually doesn't need torch except for some specific augmentations pull-ins
def mock_package(name, attrs=None):
    m = types.ModuleType(name)
    m.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    sys.modules[name] = m
    return m


mock_package("torch")
mock_package("torch.nn")  # Just in case
mock_package("torch.utils")
mock_package("torch.utils.data")

# Add PaddleOCR to path
__dir__ = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(__dir__, ".."))
paddle_ocr_path = os.path.join(workspace_root, "PaddleOCR")
sys.path.append(paddle_ocr_path)
sys.path.insert(0, os.path.join(paddle_ocr_path, "ppstructure"))

from ppstructure.layout.predict_layout import LayoutPredictor
from ppstructure.table.predict_structure import TableStructurer


class DummyArgs:
    def __init__(self, **kwargs):
        # Default Required by PaddleOCR inference core
        self.use_gpu = False
        self.use_xpu = False
        self.use_npu = False
        self.use_mlu = False
        self.use_metax_gpu = False
        self.use_gcu = False
        self.gpu_id = 0
        self.ir_optim = True
        self.use_tensorrt = False
        self.min_subgraph_size = 15
        self.precision = "fp32"
        self.gpu_mem = 500
        self.image_dir = None
        self.det_limit_side_len = 960
        self.det_limit_type = "max"
        self.det_db_thresh = 0.3
        self.det_db_box_thresh = 0.6
        self.det_db_unclip_ratio = 1.5
        self.max_batch_size = 10
        self.use_dilation = False
        self.det_db_score_mode = "fast"
        self.layout_model_dir = None
        self.layout_dict_path = os.path.join(
            paddle_ocr_path, "ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt"
        )
        self.layout_score_threshold = 0.5
        self.layout_nms_threshold = 0.5
        self.table_model_dir = None
        self.table_char_dict_path = os.path.join(
            paddle_ocr_path, "ppocr/utils/dict/table_structure_dict_ch.txt"
        )
        self.table_max_len = 488
        self.table_algorithm = "TableAttn"
        self.merge_no_span_structure = True
        self.use_onnx = False
        self.onnx_sess_options = None
        self.onnx_providers = None
        self.show_log = False
        self.benchmark = False
        self.enable_mkldnn = False
        self.cpu_threads = 10
        self.use_pdf2docx_api = False
        self.recovery = False
        self.use_mp = False
        self.total_process_num = 1
        self.process_id = 0
        self.return_word_box = False

        # Overwrite defaults with provided kwargs
        self.__dict__.update(kwargs)


def main():
    test_images = [
        os.path.join(workspace_root, "public/assets/test.png"),
        os.path.join(
            paddle_ocr_path,
            "docs/images/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808-20240708082238739.jpg",
        ),
    ]
    output_dir = os.path.join(workspace_root, "_temp_structure_v3_test")
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing Layout Predictor...")
    layout_args = DummyArgs(
        layout_model_dir=os.path.join(
            workspace_root, "_temp_structure_v3_conversion/layout"
        )
    )
    layout_predictor = LayoutPredictor(layout_args)

    print("Initializing Table Structurer...")
    table_args = DummyArgs(
        table_model_dir=os.path.join(
            workspace_root, "_temp_structure_v3_conversion/table"
        )
    )
    table_structurer = TableStructurer(table_args)

    for img_path in test_images:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path}, not found.")
            continue

        print(f"\nProcessing {os.path.basename(img_path)}...")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue

        # 1. Run Layout
        layout_res, elapse = layout_predictor(img)
        print(f"Layout took {elapse:.3f}s. Found {len(layout_res)} regions.")

        # Draw results
        vis_img = img.copy()
        for i, region in enumerate(layout_res):
            x1, y1, x2, y2 = region["bbox"]
            label = region["label"]
            score = region["score"]
            print(f"  Region {i}: {label} ({score:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

            # Draw box
            cv2.rectangle(
                vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )
            cv2.putText(
                vis_img,
                f"{label} {score:.2f}",
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # 2. Run Table if table
            if label == "table":
                roi_img = img[int(y1) : int(y2), int(x1) : int(x2)]
                if roi_img.size == 0:
                    continue

                (structure_str_list, bbox_list), table_elapse = table_structurer(
                    roi_img
                )
                print(
                    f"    Table structure analysis took {table_elapse:.3f}s. Found {len(bbox_list)} cells."
                )

                # Draw cell bboxes on a separate ROI visualization
                roi_vis = roi_img.copy()
                for cell_box in bbox_list:
                    # cell_box is [x1, y1, x2, y2]
                    cv2.rectangle(
                        roi_vis,
                        (int(cell_box[0]), int(cell_box[1])),
                        (int(cell_box[2]), int(cell_box[3])),
                        (255, 0, 0),
                        1,
                    )

                roi_save_path = os.path.join(
                    output_dir, f"{os.path.basename(img_path)}_table_{i}_cells.jpg"
                )
                cv2.imwrite(roi_save_path, roi_vis)
                print(f"    Saved table cells visualization to {roi_save_path}")

        save_path = os.path.join(output_dir, f"vis_{os.path.basename(img_path)}")
        cv2.imwrite(save_path, vis_img)
        print(f"Saved layout visualization to {save_path}")


if __name__ == "__main__":
    main()
