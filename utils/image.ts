import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";

const FIXED_IMG_SIZE = 224;
const IMAGE_MEAN = 0.5;
const IMAGE_STD = 0.5;

function trimWhiteBorder(image: cv.Mat): cv.Mat {
  const bg = new cv.Mat(image.rows, image.cols, image.type(), new cv.Scalar(255, 255, 255, 255));
  const diff = new cv.Mat();
  cv.absdiff(image, bg, diff);
  const mask = new cv.Mat();
  cv.cvtColor(diff, mask, cv.COLOR_BGR2GRAY);
  const thres = new cv.Mat();
  cv.threshold(mask, thres, 15, 255, cv.THRESH_BINARY);
  const rect = cv.boundingRect(thres);
  diff.delete();
  mask.delete();
  thres.delete();
  return image.roi(rect);
}

function padding(image: cv.Mat, requiredSize: number): cv.Mat {
  const padded = new cv.Mat();
  cv.copyMakeBorder(image, padded, 0, requiredSize - image.rows, 0, requiredSize - image.cols, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
  return padded;
}

export function transform(image: cv.Mat): Tensor {
  let trimmed = trimWhiteBorder(image);
  cv.cvtColor(trimmed, trimmed, cv.COLOR_BGR2GRAY);
  const dsize = new cv.Size(FIXED_IMG_SIZE - 1, FIXED_IMG_SIZE - 1);
  cv.resize(trimmed, trimmed, dsize, 0, 0, cv.INTER_CUBIC);
  const padded = padding(trimmed, FIXED_IMG_SIZE);
  trimmed.delete();
  
  const float32Data = new Float32Array(padded.data.length);
  for (let i = 0; i < padded.data.length; i++) {
    float32Data[i] = (padded.data[i] / 255.0 - IMAGE_MEAN) / IMAGE_STD;
  }
  
  const tensor = new Tensor("float32", float32Data, [1, 1, FIXED_IMG_SIZE, FIXED_IMG_SIZE]);
  padded.delete();
  return tensor;
}
