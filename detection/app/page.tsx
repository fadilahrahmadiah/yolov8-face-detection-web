"use client";

import { useEffect, useRef, useState } from "react";
import Script from "next/script"; 

declare const ort: any; 

export default function FaceAiPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<any>(null);
  const [modelStatus, setModelStatus] = useState<string>("Loading Model...");
  const [detectingStatus, setDetectingStatus] = useState<string>("Deteksi: Berhenti");
  const [scriptLoaded, setScriptLoaded] = useState<boolean>(false);
  const intervalId = useRef<any>(null);

  const handleScriptLoad = async () => {
    setScriptLoaded(true);
    try {
      console.log("Script ONNX siap. Memulai inisialisasi model...");
      
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/";
      ort.env.wasm.numThreads = 1;

      const sess = await ort.InferenceSession.create("/face_classification.onnx");
      setSession(sess);
      setModelStatus("Model Ready");
      console.log("SUKSES: Model ONNX berhasil dimuat!");
    } catch (e) {
      console.error("Gagal load model:", e);
      setModelStatus("Gagal Load Model");
    }
  };

  const startDetection = async () => {
    if (!session || !videoRef.current || !canvasRef.current) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 1280, height: 720 },
        audio: false 
      });
      videoRef.current.srcObject = stream;
      setDetectingStatus("Deteksi: Memulai Kamera...");

      videoRef.current.onloadedmetadata = () => {
        if (!videoRef.current || !canvasRef.current) return;
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        setDetectingStatus("Deteksi: Aktif");
        intervalId.current = setInterval(() => runInference(), 50);
      };
    } catch (e) {
      console.error("Kamera error:", e);
      setDetectingStatus("Deteksi: Gagal Kamera");
    }
  };

  const stopDetection = () => {
    if (intervalId.current) clearInterval(intervalId.current);
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setDetectingStatus("Deteksi: Berhenti");
  };

const runInference = async () => {
    if (!session || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const videoWidth = video.videoWidth;
    const videoHeight = video.videoHeight;

    try {
      const imageSize = 640;
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = imageSize;
      tempCanvas.height = imageSize;
      const tempCtx = tempCanvas.getContext("2d");
      if (!tempCtx) return;
      tempCtx.drawImage(video, 0, 0, imageSize, imageSize);

      const imgData = tempCtx.getImageData(0, 0, imageSize, imageSize);
      const { data } = imgData;
      const input = new Float32Array(imageSize * imageSize * 3);

      for (let i = 0; i < data.length; i += 4) {
        input[i / 4] = data[i] / 255;
        input[i / 4 + imageSize * imageSize] = data[i + 1] / 255;
        input[i / 4 + imageSize * imageSize * 2] = data[i + 2] / 255;
      }

      const tensorInput = new ort.Tensor("float32", input, [1, 3, imageSize, imageSize]);
      const feeds: any = {};
      feeds[session.inputNames[0]] = tensorInput;
      const outputData = await session.run(feeds);
      
      const firstOutputName = session.outputNames[0];
      const outputTensor = outputData[firstOutputName];
      const outputRaw = outputTensor.data as Float32Array;
      const dims = outputTensor.dims; 

      const numRows = dims[1]; 
      const numCandidates = dims[2]; 

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const CONFIDENCE_THRESHOLD = 0.45; 
      let candidates: any[] = [];

      const labelKelas = ["eye", "face", "lips", "nose"]; 
      
      const warnaKelas: { [key: string]: string } = {
        "face": "#34C759", 
        "eye": "#00F0FF",  
        "nose": "#FFCC00", 
        "lips": "#FF2D55"  
      };

      for (let c = 0; c < numCandidates; c++) {
        let maxClassScore = -1;
        let classId = -1;

        for (let r = 4; r < numRows; r++) {
          const classScore = outputRaw[r * numCandidates + c];
          if (classScore > maxClassScore) {
            maxClassScore = classScore;
            classId = r - 4; 
          }
        }

        if (maxClassScore > CONFIDENCE_THRESHOLD) {
          const cx = outputRaw[0 * numCandidates + c];
          const cy = outputRaw[1 * numCandidates + c];
          const w = outputRaw[2 * numCandidates + c];
          const h = outputRaw[3 * numCandidates + c];

          const x1 = cx - w / 2;
          const y1 = cy - h / 2;

          const scaleX = videoWidth / imageSize;
          const scaleY = videoHeight / imageSize;

          candidates.push({
            x: x1 * scaleX,
            y: y1 * scaleY,
            width: w * scaleX,
            height: h * scaleY,
            score: maxClassScore,
            className: labelKelas[classId] || `Objek ${classId}`
          });
        }
      }

      candidates.sort((a, b) => b.score - a.score);

      const finalBoxes: any[] = [];
      const IOU_THRESHOLD = 0.45; 

      const calculateIoU = (boxA: any, boxB: any) => {
        const xA = Math.max(boxA.x, boxB.x);
        const yA = Math.max(boxA.y, boxB.y);
        const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
        const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

        const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
        if (interArea === 0) return 0;

        const boxAArea = boxA.width * boxA.height;
        const boxBArea = boxB.width * boxB.height;
        return interArea / (boxAArea + boxBArea - interArea);
      };

      while (candidates.length > 0) {
        const bestBox = candidates.shift(); 
        finalBoxes.push(bestBox);

        candidates = candidates.filter((box) => {
          if (box.className !== bestBox.className) return true; 
          const iou = calculateIoU(bestBox, box);
          return iou < IOU_THRESHOLD; 
        });
      }

      finalBoxes.forEach((box) => {
        const warna = warnaKelas[box.className] || "#FFFFFF";

        ctx.strokeStyle = warna;
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        ctx.fillStyle = warna;
        ctx.fillRect(box.x, box.y - 25, 100, 25);

        ctx.font = "bold 14px Arial";
        ctx.fillStyle = "#000000"; 
        ctx.fillText(`${box.className} ${(box.score * 100).toFixed(0)}%`, box.x + 6, box.y - 7);
      });

    } catch (e) {
      console.error("Inference Error:", e);
    }
  };

  return (
    <div className="flex flex-col items-center p-6 bg-slate-900 min-h-screen text-slate-100">
      <Script 
        src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js"
        strategy="afterInteractive"
        onLoad={handleScriptLoad} 
      />

      <h1 className="text-3xl font-bold mb-6 text-center text-teal-300">
        Face Detection Dashboard
      </h1>
      
      <div className="flex gap-4 mb-6">
        <div className="p-4 bg-slate-800 rounded-lg shadow">
          <p className="text-sm text-slate-400">Status Model:</p>
          <p className="text-xl font-mono font-bold">{modelStatus}</p>
        </div>
        <div className="p-4 bg-slate-800 rounded-lg shadow">
          <p className="text-sm text-slate-400">Status Deteksi:</p>
          <p className="text-xl font-mono font-bold text-amber-400">{detectingStatus}</p>
        </div>
      </div>

      <div className="relative border-4 border-slate-700 rounded-2xl shadow-xl bg-black overflow-hidden aspect-video w-full max-w-4xl">
        <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 w-full h-full object-cover" />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-10" />
      </div>

      <div className="flex gap-4 mt-6">
        <button 
          onClick={startDetection} 
          disabled={!session} 
          className="px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition disabled:bg-slate-600 disabled:opacity-50 font-semibold"
        >
          Mulai Deteksi Wajah
        </button>
        <button 
          onClick={stopDetection} 
          className="px-6 py-3 bg-amber-600 text-white rounded-lg hover:bg-amber-700 transition font-semibold"
        >
          Stop Deteksi
        </button>
      </div>
    </div>
  );
}