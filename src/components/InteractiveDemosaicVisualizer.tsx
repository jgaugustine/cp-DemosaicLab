import { useEffect, useRef, useState, useMemo } from "react";
import { Card } from "@/components/ui/card";
import { DemosaicInput, DemosaicAlgorithm, DemosaicParams } from "@/types/demosaic";
import { getBayerKernel, getXTransKernel } from "@/lib/cfa";
import { 
  demosaicNearest,
  demosaicBilinear,
  demosaicNiuEdgeSensing,
  demosaicLienEdgeBased,
  demosaicWuPolynomial,
  demosaicKikuResidual,
  demosaicXTransNiuEdgeSensing,
  demosaicXTransLienEdgeBased,
  demosaicXTransWuPolynomial,
  demosaicXTransKikuResidual,
} from "@/lib/demosaic";
import { InlineMath } from "react-katex";
import "katex/dist/katex.min.css";

interface InteractiveDemosaicVisualizerProps {
  input: DemosaicInput;
  centerX: number;
  centerY: number;
  algorithm: DemosaicAlgorithm;
  params?: DemosaicParams;
}

const REGION_SIZE = 15; // Odd number for easier centering

// Helper functions for advanced algorithms
const logisticFunction = (x: number, threshold: number = 0.1, steepness?: number): number => {
  const k = steepness !== undefined 
    ? steepness 
    : 20.0 / Math.max(0.01, threshold);
  return 1.0 / (1.0 + Math.exp(-k * (x - threshold)));
};

const computeDirectionalVariations = (
  cfaData: Float32Array | Uint16Array,
  width: number,
  height: number,
  x: number,
  y: number,
  getChannel: (x: number, y: number) => 'r' | 'g' | 'b',
  getVal: (x: number, y: number) => number
): { horizontal: number; vertical: number } => {
  const hVar = Math.abs(getVal(x + 1, y) - getVal(x - 1, y));
  const vVar = Math.abs(getVal(x, y + 1) - getVal(x, y - 1));
  return { horizontal: hVar, vertical: vVar };
};

// Helper to collect neighbors of a specific color with expanding search radius
const collectNeighbors = (
  cfaData: Float32Array | Uint16Array,
  width: number,
  height: number,
  x: number,
  y: number,
  targetColor: 'r' | 'g' | 'b',
  getChannel: (x: number, y: number) => 'r' | 'g' | 'b',
  getVal: (x: number, y: number) => number,
  maxRadius: number = 10
): { values: number[]; distances: number[] } => {
  const result: { values: number[]; distances: number[] } = { values: [], distances: [] };
  
  // Collect all neighbors within maxRadius
  for (let dy = -maxRadius; dy <= maxRadius; dy++) {
    for (let dx = -maxRadius; dx <= maxRadius; dx++) {
      if (dx === 0 && dy === 0) continue;
      const ch = getChannel(x + dx, y + dy);
      if (ch === targetColor) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        result.values.push(getVal(x + dx, y + dy));
        result.distances.push(dist);
      }
    }
  }
  
  return result;
};

// Polynomial interpolation helper
const polynomialInterpolate = (
  values: number[],
  distances: number[],
  degree: number = 2
): number => {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];
  
  if (degree === 1) {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  
  const weights = distances.map((d) => {
    const dist = Math.max(0.1, d);
    return 1.0 / (1.0 + Math.pow(dist, degree));
  });
  
  const sumW = weights.reduce((a, b) => a + b, 0);
  if (sumW === 0) {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  
  return values.reduce((sum, v, i) => sum + v * weights[i], 0) / sumW;
};

export function InteractiveDemosaicVisualizer({
  input,
  centerX,
  centerY,
  algorithm,
  params,
}: InteractiveDemosaicVisualizerProps) {
  const inputCanvasRef = useRef<HTMLCanvasElement>(null);
  const inputOverlayRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputOverlayRef = useRef<HTMLCanvasElement>(null);
  
  // Local cursor position within the region (0 to REGION_SIZE-1)
  // Default to center
  const [localCursorX, setLocalCursorX] = useState(Math.floor(REGION_SIZE / 2));
  const [localCursorY, setLocalCursorY] = useState(Math.floor(REGION_SIZE / 2));
  const [isDragging, setIsDragging] = useState(false);

  // Calculate the top-left of the region in global coordinates
  const regionOriginX = centerX - Math.floor(REGION_SIZE / 2);
  const regionOriginY = centerY - Math.floor(REGION_SIZE / 2);

  // Helper to get global coordinates from local cursor
  const globalCursorX = regionOriginX + localCursorX;
  const globalCursorY = regionOriginY + localCursorY;

  // Extract region data (CFA and Demosaiced)
  // Use the actual demosaic functions and extract the region from the full output
  const regionData = useMemo(() => {
    const { width, height, cfaData, cfaPatternMeta, cfaPattern } = input;
    const inputImageData = new ImageData(REGION_SIZE, REGION_SIZE);
    const outputImageData = new ImageData(REGION_SIZE, REGION_SIZE);
    
    // Run the actual demosaic algorithm on the full image
    let fullOutput: ImageData;
    if (input.cfaPattern === 'xtrans') {
      switch (algorithm) {
        case 'nearest':
          fullOutput = demosaicNearest(input);
          break;
        case 'bilinear':
          fullOutput = demosaicBilinear(input);
          break;
        case 'niu_edge_sensing':
          fullOutput = demosaicXTransNiuEdgeSensing(input, params);
          break;
        case 'lien_edge_based':
          fullOutput = demosaicXTransLienEdgeBased(input);
          break;
        case 'wu_polynomial':
          fullOutput = demosaicXTransWuPolynomial(input, params);
          break;
        case 'kiku_residual':
          fullOutput = demosaicXTransKikuResidual(input, params);
          break;
        default:
          fullOutput = new ImageData(width, height);
      }
    } else {
      switch (algorithm) {
        case 'nearest':
          fullOutput = demosaicNearest(input);
          break;
        case 'bilinear':
          fullOutput = demosaicBilinear(input);
          break;
        case 'niu_edge_sensing':
          fullOutput = demosaicNiuEdgeSensing(input, params);
          break;
        case 'lien_edge_based':
          fullOutput = demosaicLienEdgeBased(input);
          break;
        case 'wu_polynomial':
          fullOutput = demosaicWuPolynomial(input, params);
          break;
        case 'kiku_residual':
          fullOutput = demosaicKikuResidual(input, params);
          break;
        default:
          fullOutput = new ImageData(width, height);
      }
    }
    
    // Fix: Properly handle both Bayer and XTrans patterns
    const getChannel = cfaPattern === 'bayer' 
      ? getBayerKernel(cfaPatternMeta.layout) 
      : cfaPattern === 'xtrans'
      ? getXTransKernel()
      : (x: number, y: number) => 'g' as const; // Fallback only for unknown patterns
    const getVal = (x: number, y: number) => {
        // Mirror padding
        let sx = x; 
        if (sx < 0) sx = -sx; 
        else if (sx >= width) sx = 2*width - 2 - sx;
        
        let sy = y; 
        if (sy < 0) sy = -sy; 
        else if (sy >= height) sy = 2*height - 2 - sy;
        
        sx = Math.max(0, Math.min(width - 1, sx));
        sy = Math.max(0, Math.min(height - 1, sy));
        return cfaData[sy * width + sx];
    };

    // Generate Input Mosaic Visualization for Region and extract output region
    for (let y = 0; y < REGION_SIZE; y++) {
      for (let x = 0; x < REGION_SIZE; x++) {
        const gx = regionOriginX + x;
        const gy = regionOriginY + y;
        
        const idx = (y * REGION_SIZE + x) * 4;
        
        // Check bounds for drawing black outside
        if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
           inputImageData.data[idx] = 0;
           inputImageData.data[idx+1] = 0;
           inputImageData.data[idx+2] = 0;
           inputImageData.data[idx+3] = 255;
           outputImageData.data[idx] = 0;
           outputImageData.data[idx+1] = 0;
           outputImageData.data[idx+2] = 0;
           outputImageData.data[idx+3] = 255;
           continue;
        }

        // Extract CFA input visualization
        const val = getVal(gx, gy);
        const ch = getChannel(gx, gy);
        const v = Math.round(val * 255);
        
        inputImageData.data[idx] = ch === 'r' ? v : 0;
        inputImageData.data[idx+1] = ch === 'g' ? v : 0;
        inputImageData.data[idx+2] = ch === 'b' ? v : 0;
        inputImageData.data[idx+3] = 255;
        
        // Extract demosaiced output from full image
        const fullIdx = (gy * width + gx) * 4;
        outputImageData.data[idx] = fullOutput.data[fullIdx];
        outputImageData.data[idx+1] = fullOutput.data[fullIdx+1];
        outputImageData.data[idx+2] = fullOutput.data[fullIdx+2];
        outputImageData.data[idx+3] = fullOutput.data[fullIdx+3];
      }
    }
    
    return { inputImageData, outputImageData };
  }, [input, regionOriginX, regionOriginY, algorithm, params]);

  // Compute Kernels for the current cursor position
  const kernels = useMemo(() => {
    const gx = globalCursorX;
    const gy = globalCursorY;
    const { width, height, cfaPatternMeta, cfaPattern } = input;
    const getChannel = cfaPattern === 'bayer' 
      ? getBayerKernel(cfaPatternMeta.layout) 
      : cfaPattern === 'xtrans'
      ? getXTransKernel()
      : (x: number, y: number) => 'g' as const;
    const centerCh = getChannel(gx, gy);
    
    // Define a small window around cursor to show weights, e.g., 3x3
    const kSize = 3;
    const offset = 1; // (kSize - 1) / 2

    // Initialize 3x3 grids for R, G, B weights
    const wR = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
    const wG = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
    const wB = Array(kSize).fill(0).map(() => Array(kSize).fill(0));
    
    if (algorithm === 'nearest') {
        if (centerCh === 'r') {
            wR[1][1] = 1;
            wG[1][2] = 1; // Right neighbor
            wB[2][2] = 1; // Bottom-Right neighbor
        } else if (centerCh === 'b') {
            wB[1][1] = 1;
            wG[1][0] = 1; // Left neighbor
            wR[0][0] = 1; // Top-Left neighbor
        } else { // Green
             wG[1][1] = 1;
             const leftCh = getChannel(gx - 1, gy);
             const rightCh = getChannel(gx + 1, gy);
             const isRedRow = (leftCh === 'r' || rightCh === 'r');
             if (isRedRow) {
                 wR[1][2] = 1; // Right
                 wB[2][1] = 1; // Bottom
             } else {
                 wB[1][2] = 1; // Right
                 wR[2][1] = 1; // Bottom
             }
        }
    } else if (algorithm === 'bilinear') {
        if (centerCh === 'g') {
            wG[1][1] = 1; // Identity
            // Red neighbors - check actual positions
            const redNeighbors = [
                { x: -1, y: 0 }, { x: 1, y: 0 }, { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'r';
            });
            // Blue neighbors - check actual positions
            const blueNeighbors = [
                { x: -1, y: 0 }, { x: 1, y: 0 }, { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'b';
            });
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            if (isRedRow) {
                // Red neighbors are Left/Right
                const redLR = redNeighbors.filter(pos => pos.x !== 0);
                const redWeight = redLR.length > 0 ? 1.0 / redLR.length : 0;
                redLR.forEach(pos => {
                    wR[1 + pos.y][1 + pos.x] = redWeight;
                });
                // Blue neighbors are Top/Bottom
                const blueTB = blueNeighbors.filter(pos => pos.y !== 0);
                const blueWeight = blueTB.length > 0 ? 1.0 / blueTB.length : 0;
                blueTB.forEach(pos => {
                    wB[1 + pos.y][1 + pos.x] = blueWeight;
                });
            } else {
                // Blue neighbors are Left/Right
                const blueLR = blueNeighbors.filter(pos => pos.x !== 0);
                const blueWeight = blueLR.length > 0 ? 1.0 / blueLR.length : 0;
                blueLR.forEach(pos => {
                    wB[1 + pos.y][1 + pos.x] = blueWeight;
                });
                // Red neighbors are Top/Bottom
                const redTB = redNeighbors.filter(pos => pos.y !== 0);
                const redWeight = redTB.length > 0 ? 1.0 / redTB.length : 0;
                redTB.forEach(pos => {
                    wR[1 + pos.y][1 + pos.x] = redWeight;
                });
            }
        } else if (centerCh === 'r') {
            wR[1][1] = 1;
            // Green: Cross average - only count actual green pixels
            const greenCross = [
                { x: -1, y: 0 }, { x: 1, y: 0 },
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenWeight = greenCross.length > 0 ? 1.0 / greenCross.length : 0;
            greenCross.forEach(pos => {
                wG[1 + pos.y][1 + pos.x] = greenWeight;
            });
            // Blue: Only set weights at corners that are actually blue in the CFA
            const blueCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'b';
            });
            const blueWeight = blueCorners.length > 0 ? 1.0 / blueCorners.length : 0;
            blueCorners.forEach(pos => {
                wB[1 + pos.y][1 + pos.x] = blueWeight;
            });
        } else { // Blue
            wB[1][1] = 1;
            // Green: Cross average - only count actual green pixels
            const greenCross = [
                { x: -1, y: 0 }, { x: 1, y: 0 },
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenWeight = greenCross.length > 0 ? 1.0 / greenCross.length : 0;
            greenCross.forEach(pos => {
                wG[1 + pos.y][1 + pos.x] = greenWeight;
            });
            // Red: Only set weights at corners that are actually red in the CFA
            const redCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'r';
            });
            const redWeight = redCorners.length > 0 ? 1.0 / redCorners.length : 0;
            redCorners.forEach(pos => {
                wR[1 + pos.y][1 + pos.x] = redWeight;
            });
        }
    } else if (algorithm === 'niu_edge_sensing') {
        // Similar to bilinear but weights depend on edge direction
        const threshold = params?.niuLogisticThreshold ?? 0.1;
        const steepness = params?.niuLogisticSteepness;
        const getVal = (x: number, y: number) => {
          if (x < 0 || x >= input.width || y < 0 || y >= input.height) return 0;
          return input.cfaData[y * input.width + x];
        };
        
        if (centerCh === 'r') {
            wR[1][1] = 1;
            // Green: weighted average of cross (edge-aware) - only count actual green pixels
            const greenHorizontal = [
                { x: -1, y: 0 }, { x: 1, y: 0 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenVertical = [
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const vars = computeDirectionalVariations(input.cfaData, input.width, input.height, gx, gy, getChannel, getVal);
            const wH = logisticFunction(vars.horizontal, threshold, steepness);
            const wV = logisticFunction(vars.vertical, threshold, steepness);
            const sumW = wH + wV;
            const nH = (sumW > 0) ? (1.0 - wH / sumW) : 0.5;
            const nV = (sumW > 0) ? (1.0 - wV / sumW) : 0.5;
            const totalWeight = nH + nV;
            if (totalWeight > 0 && greenHorizontal.length > 0) {
                const hWeight = (nH / totalWeight) / greenHorizontal.length;
                greenHorizontal.forEach(pos => {
                    wG[1 + pos.y][1 + pos.x] = hWeight;
                });
            }
            if (totalWeight > 0 && greenVertical.length > 0) {
                const vWeight = (nV / totalWeight) / greenVertical.length;
                greenVertical.forEach(pos => {
                    wG[1 + pos.y][1 + pos.x] = vWeight;
                });
            }
            // Blue: Only set weights at corners that are actually blue in the CFA
            const blueCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'b';
            });
            const blueWeight = blueCorners.length > 0 ? 1.0 / blueCorners.length : 0;
            blueCorners.forEach(pos => {
                wB[1 + pos.y][1 + pos.x] = blueWeight;
            });
        } else if (centerCh === 'b') {
            wB[1][1] = 1;
            // Green: weighted average of cross (edge-aware) - only count actual green pixels
            const greenHorizontal = [
                { x: -1, y: 0 }, { x: 1, y: 0 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenVertical = [
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const vars = computeDirectionalVariations(input.cfaData, input.width, input.height, gx, gy, getChannel, getVal);
            const wH = logisticFunction(vars.horizontal, threshold, steepness);
            const wV = logisticFunction(vars.vertical, threshold, steepness);
            const sumW = wH + wV;
            const nH = (sumW > 0) ? (1.0 - wH / sumW) : 0.5;
            const nV = (sumW > 0) ? (1.0 - wV / sumW) : 0.5;
            const totalWeight = nH + nV;
            if (totalWeight > 0 && greenHorizontal.length > 0) {
                const hWeight = (nH / totalWeight) / greenHorizontal.length;
                greenHorizontal.forEach(pos => {
                    wG[1 + pos.y][1 + pos.x] = hWeight;
                });
            }
            if (totalWeight > 0 && greenVertical.length > 0) {
                const vWeight = (nV / totalWeight) / greenVertical.length;
                greenVertical.forEach(pos => {
                    wG[1 + pos.y][1 + pos.x] = vWeight;
                });
            }
            // Red: Only set weights at corners that are actually red in the CFA
            const redCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'r';
            });
            const redWeight = redCorners.length > 0 ? 1.0 / redCorners.length : 0;
            redCorners.forEach(pos => {
                wR[1 + pos.y][1 + pos.x] = redWeight;
            });
        } else { // Green
            wG[1][1] = 1;
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            if (isRedRow) {
                wR[1][0] = 0.5; wR[1][2] = 0.5;
                wB[0][1] = 0.5; wB[2][1] = 0.5;
            } else {
                wB[1][0] = 0.5; wB[1][2] = 0.5;
                wR[0][1] = 0.5; wR[2][1] = 0.5;
            }
        }
    } else if (algorithm === 'lien_edge_based') {
        // Similar to bilinear but with edge-aware interpolation
        if (centerCh === 'g') {
            wG[1][1] = 1;
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            // Edge direction determines which neighbors to use (simplified: use bilinear pattern)
            if (isRedRow) {
                wR[1][0] = 0.5; wR[1][2] = 0.5;
                wB[0][1] = 0.5; wB[2][1] = 0.5;
            } else {
                wB[1][0] = 0.5; wB[1][2] = 0.5;
                wR[0][1] = 0.5; wR[2][1] = 0.5;
            }
        } else if (centerCh === 'r') {
            wR[1][1] = 1;
            // Edge-aware green (simplified: use cross average) - only count actual green pixels
            const greenCross = [
                { x: -1, y: 0 }, { x: 1, y: 0 },
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenWeight = greenCross.length > 0 ? 1.0 / greenCross.length : 0;
            greenCross.forEach(pos => {
                wG[1 + pos.y][1 + pos.x] = greenWeight;
            });
            // Blue: Only set weights at corners that are actually blue in the CFA
            const blueCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'b';
            });
            const blueWeight = blueCorners.length > 0 ? 1.0 / blueCorners.length : 0;
            blueCorners.forEach(pos => {
                wB[1 + pos.y][1 + pos.x] = blueWeight;
            });
        } else {
            wB[1][1] = 1;
            // Green: Cross average - only count actual green pixels
            const greenCross = [
                { x: -1, y: 0 }, { x: 1, y: 0 },
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenWeight = greenCross.length > 0 ? 1.0 / greenCross.length : 0;
            greenCross.forEach(pos => {
                wG[1 + pos.y][1 + pos.x] = greenWeight;
            });
            // Red: Only set weights at corners that are actually red in the CFA
            const redCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'r';
            });
            const redWeight = redCorners.length > 0 ? 1.0 / redCorners.length : 0;
            redCorners.forEach(pos => {
                wR[1 + pos.y][1 + pos.x] = redWeight;
            });
        }
    } else if (algorithm === 'wu_polynomial' || algorithm === 'kiku_residual') {
        // These use polynomial/residual refinement, but for visualization use bilinear pattern
        if (centerCh === 'g') {
            wG[1][1] = 1;
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            if (isRedRow) {
                wR[1][0] = 0.5; wR[1][2] = 0.5;
                wB[0][1] = 0.5; wB[2][1] = 0.5;
            } else {
                wB[1][0] = 0.5; wB[1][2] = 0.5;
                wR[0][1] = 0.5; wR[2][1] = 0.5;
            }
        } else if (centerCh === 'r') {
            wR[1][1] = 1;
            // Green: Cross average - only count actual green pixels
            const greenCross = [
                { x: -1, y: 0 }, { x: 1, y: 0 },
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenWeight = greenCross.length > 0 ? 1.0 / greenCross.length : 0;
            greenCross.forEach(pos => {
                wG[1 + pos.y][1 + pos.x] = greenWeight;
            });
            // Blue: Only set weights at corners that are actually blue in the CFA
            const blueCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'b';
            });
            const blueWeight = blueCorners.length > 0 ? 1.0 / blueCorners.length : 0;
            blueCorners.forEach(pos => {
                wB[1 + pos.y][1 + pos.x] = blueWeight;
            });
        } else {
            wB[1][1] = 1;
            // Green: Cross average - only count actual green pixels
            const greenCross = [
                { x: -1, y: 0 }, { x: 1, y: 0 },
                { x: 0, y: -1 }, { x: 0, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'g';
            });
            const greenWeight = greenCross.length > 0 ? 1.0 / greenCross.length : 0;
            greenCross.forEach(pos => {
                wG[1 + pos.y][1 + pos.x] = greenWeight;
            });
            // Red: Only set weights at corners that are actually red in the CFA
            const redCorners = [
                { x: -1, y: -1 }, { x: 1, y: -1 },
                { x: -1, y: 1 }, { x: 1, y: 1 }
            ].filter(pos => {
                const px = gx + pos.x;
                const py = gy + pos.y;
                return px >= 0 && px < input.width && py >= 0 && py < input.height && getChannel(px, py) === 'r';
            });
            const redWeight = redCorners.length > 0 ? 1.0 / redCorners.length : 0;
            redCorners.forEach(pos => {
                wR[1 + pos.y][1 + pos.x] = redWeight;
            });
        }
    }
    
    return { wR, wG, wB, kSize };
  }, [globalCursorX, globalCursorY, algorithm, input]);

  // Generate Visualization Data for Diagrams
  const kernelVisualizations = useMemo(() => {
    const { wR, wG, wB, kSize } = kernels;
    const offset = Math.floor(kSize / 2);
    const visualizations = [];
    
    // Get channel function once (reuse it for consistency)
    const getChannel = input.cfaPattern === 'bayer' 
      ? getBayerKernel(input.cfaPatternMeta.layout) 
      : input.cfaPattern === 'xtrans'
      ? getXTransKernel()
      : (x: number, y: number) => 'g' as const;
    
    const makeVis = (weights: number[][], title: string, targetChannel: 'r' | 'g' | 'b') => {
        const cells = [];
        let totalR = 0, totalG = 0, totalB = 0;
        
        for (let y = 0; y < kSize; y++) {
            for (let x = 0; x < kSize; x++) {
                const weight = weights[y][x];
                // Only include cells with non-zero weights
                if (Math.abs(weight) < 1e-6) continue;
                
                // Get pixel value from input
                const gx = globalCursorX + (x - offset);
                const gy = globalCursorY + (y - offset);
                
                let r = 0, g = 0, b = 0;
                let label = "";
                let ch: 'r' | 'g' | 'b' = 'g'; // Default
                // Safely get mosaic value
                if (gx >= 0 && gx < input.width && gy >= 0 && gy < input.height) {
                     const val = input.cfaData[gy * input.width + gx] * 255;
                     const clampedVal = Math.max(0, Math.min(255, Math.round(val)));
                     // Use the getChannel function defined above
                     const channelResult = getChannel(gx, gy);
                     ch = (channelResult === 'r' || channelResult === 'g' || channelResult === 'b') 
                         ? channelResult 
                         : 'g';
                     
                     // Set RGB values based on CFA channel - only the actual channel has value
                     if (ch === 'r') { 
                         r = clampedVal; 
                         g = 0;
                         b = 0;
                         label = `R: ${clampedVal}`;
                     }
                     else if (ch === 'g') { 
                         r = 0;
                         g = clampedVal; 
                         b = 0;
                         label = `G: ${clampedVal}`;
                     }
                     else { 
                         r = 0;
                         g = 0;
                         b = clampedVal; 
                         label = `B: ${clampedVal}`;
                     }
                } else {
                    // Out of bounds - keep r, g, b as 0
                }
                
                // For channel reconstruction diagrams, only show pixels that match the target channel
                // This ensures we only display the actual contributing pixels for each channel
                if (targetChannel === 'r' && ch !== 'r') {
                    continue; // Skip non-red pixels for red channel reconstruction
                } else if (targetChannel === 'g' && ch !== 'g') {
                    continue; // Skip non-green pixels for green channel reconstruction
                } else if (targetChannel === 'b' && ch !== 'b') {
                    continue; // Skip non-blue pixels for blue channel reconstruction
                }
                
                cells.push({ r, g, b, weight, label });
                
                // Accumulate contributions based on the target channel being reconstructed
                // The weight tells us how much this pixel contributes to the target channel
                // Each pixel has a specific CFA channel (r, g, or b) which we use
                if (targetChannel === 'r') {
                    // For red reconstruction, only red pixels contribute to red
                    // Green and blue pixels are shown but don't contribute to red channel output
                    totalR += r * weight;
                } else if (targetChannel === 'g') {
                    // For green reconstruction, only green pixels contribute to green
                    totalG += g * weight;
                } else {
                    // For blue reconstruction, only blue pixels contribute to blue
                    totalB += b * weight;
                }
            }
        }
        
        // Output should only show the contribution to the target channel
        // Other channels should be 0 since this is a single-channel reconstruction
        const totals = targetChannel === 'r' 
            ? { r: totalR, g: 0, b: 0 }
            : targetChannel === 'g'
            ? { r: 0, g: totalG, b: 0 }
            : { r: 0, g: 0, b: totalB };
        
        // Return filtered cells (only non-zero weights)
        return { title, size: cells.length, cells, totals };
    };
    
    visualizations.push(makeVis(wR, "Red Channel Reconstruction", 'r'));
    visualizations.push(makeVis(wG, "Green Channel Reconstruction", 'g'));
    visualizations.push(makeVis(wB, "Blue Channel Reconstruction", 'b'));
    
    return visualizations;
  }, [kernels, globalCursorX, globalCursorY, input]);

  // Compute which pixels contribute to each channel for arrow visualization
  const contributingPixels = useMemo(() => {
    const gx = globalCursorX;
    const gy = globalCursorY;
    const { width, height, cfaPatternMeta, cfaPattern } = input;
    const getChannel = cfaPattern === 'bayer' 
      ? getBayerKernel(cfaPatternMeta.layout) 
      : cfaPattern === 'xtrans'
      ? getXTransKernel()
      : (x: number, y: number) => 'g' as const;
    
    const centerCh = getChannel(gx, gy);
    const getVal = (x: number, y: number) => {
      if (x < 0 || x >= width || y < 0 || y >= height) return 0;
      return input.cfaData[y * width + x];
    };
    
    const rPixels: Array<{x: number, y: number, dominant?: boolean}> = [];
    const gPixels: Array<{x: number, y: number, dominant?: boolean}> = [];
    const bPixels: Array<{x: number, y: number, dominant?: boolean}> = [];
    
    // Helper function to collect neighbors
    const collectNeighbors = (targetColor: 'r' | 'g' | 'b', maxRadius: number = 10) => {
        const result: Array<{x: number, y: number, dominant?: boolean}> = [];
        for (let dy = -maxRadius; dy <= maxRadius; dy++) {
            for (let dx = -maxRadius; dx <= maxRadius; dx++) {
                if (dx === 0 && dy === 0) continue;
                const px = gx + dx;
                const py = gy + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    const ch = getChannel(px, py);
                    if (ch === targetColor) {
                        result.push({x: px, y: py});
                    }
                }
            }
        }
        return result;
    };
    
    // Add center pixel to appropriate channel
    if (centerCh === 'r') rPixels.push({x: gx, y: gy});
    else if (centerCh === 'g') gPixels.push({x: gx, y: gy});
    else bPixels.push({x: gx, y: gy});
    
    if (algorithm === 'nearest') {
      if (centerCh === 'r') {
        // Find nearest G and B
        for (let d = 1; d <= 10; d++) {
          for (let dy = -d; dy <= d; dy++) {
            for (let dx = -d; dx <= d; dx++) {
              if (dx === 0 && dy === 0) continue;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist > d - 1 && dist <= d) {
                const ch = getChannel(gx + dx, gy + dy);
                if (ch === 'g' && gPixels.length === 0) {
                  gPixels.push({x: gx + dx, y: gy + dy});
                }
                if (ch === 'b' && bPixels.length === 0) {
                  bPixels.push({x: gx + dx, y: gy + dy});
                }
              }
            }
          }
          if (gPixels.length > 0 && bPixels.length > 0) break;
        }
      } else if (centerCh === 'b') {
        // Find nearest G and R
        for (let d = 1; d <= 10; d++) {
          for (let dy = -d; dy <= d; dy++) {
            for (let dx = -d; dx <= d; dx++) {
              if (dx === 0 && dy === 0) continue;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist > d - 1 && dist <= d) {
                const ch = getChannel(gx + dx, gy + dy);
                if (ch === 'g' && gPixels.length === 0) {
                  gPixels.push({x: gx + dx, y: gy + dy});
                }
                if (ch === 'r' && rPixels.length === 0) {
                  rPixels.push({x: gx + dx, y: gy + dy});
                }
              }
            }
          }
          if (gPixels.length > 0 && rPixels.length > 0) break;
        }
      } else {
        // Green pixel - find nearest R and B
        for (let d = 1; d <= 10; d++) {
          for (let dy = -d; dy <= d; dy++) {
            for (let dx = -d; dx <= d; dx++) {
              if (dx === 0 && dy === 0) continue;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist > d - 1 && dist <= d) {
                const ch = getChannel(gx + dx, gy + dy);
                if (ch === 'r' && rPixels.length === 0) {
                  rPixels.push({x: gx + dx, y: gy + dy});
                }
                if (ch === 'b' && bPixels.length === 0) {
                  bPixels.push({x: gx + dx, y: gy + dy});
                }
              }
            }
          }
          if (rPixels.length > 0 && bPixels.length > 0) break;
        }
      }
    } else if (algorithm === 'bilinear') {
      // Use collectNeighbors with radius 1 to match actual implementation
          if (centerCh === 'g') {
        const rNeighbors = collectNeighbors('r', 1);
        const bNeighbors = collectNeighbors('b', 1);
        rPixels.push(...rNeighbors);
        bPixels.push(...bNeighbors);
          } else if (centerCh === 'r') {
        const gNeighbors = collectNeighbors('g', 1);
        const bNeighbors = collectNeighbors('b', 1);
        gPixels.push(...gNeighbors);
        bPixels.push(...bNeighbors);
          } else { // Blue
        const gNeighbors = collectNeighbors('g', 1);
        const rNeighbors = collectNeighbors('r', 1);
        gPixels.push(...gNeighbors);
        rPixels.push(...rNeighbors);
      }
    } else if (algorithm === 'niu_edge_sensing') {
      // X-Trans uses radius 2 (5x5), Bayer uses radius 1 (3x3)
      const searchRadius = cfaPattern === 'xtrans' ? 2 : 1;
      if (centerCh === 'r') {
        if (cfaPattern === 'xtrans') {
          // X-Trans: collect horizontal and vertical green neighbors separately, and all blue neighbors
          const gNeighborsH = collectNeighbors('g', searchRadius).filter(p => p.y === gy);
          const gNeighborsV = collectNeighbors('g', searchRadius).filter(p => p.x === gx);
          const bNeighbors = collectNeighbors('b', searchRadius);
          gPixels.push(...gNeighborsH, ...gNeighborsV);
          bPixels.push(...bNeighbors);
        } else {
          // Bayer: use all green neighbors and corner blue neighbors
          const gNeighbors = collectNeighbors('g', searchRadius);
          const bCorners = collectNeighbors('b', searchRadius).filter(p => 
            (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
          );
          gPixels.push(...gNeighbors);
          bPixels.push(...bCorners);
        }
      } else if (centerCh === 'b') {
        if (cfaPattern === 'xtrans') {
          // X-Trans: collect horizontal and vertical green neighbors separately, and all red neighbors
          const gNeighborsH = collectNeighbors('g', searchRadius).filter(p => p.y === gy);
          const gNeighborsV = collectNeighbors('g', searchRadius).filter(p => p.x === gx);
          const rNeighbors = collectNeighbors('r', searchRadius);
          gPixels.push(...gNeighborsH, ...gNeighborsV);
          rPixels.push(...rNeighbors);
        } else {
          // Bayer: use all green neighbors and corner red neighbors
          const gNeighbors = collectNeighbors('g', searchRadius);
          const rCorners = collectNeighbors('r', searchRadius).filter(p => 
            (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
          );
          gPixels.push(...gNeighbors);
          rPixels.push(...rCorners);
        }
      } else { // Green
        if (cfaPattern === 'xtrans') {
          // X-Trans: collect all red and blue neighbors in 5x5
          const rNeighbors = collectNeighbors('r', searchRadius);
          const bNeighbors = collectNeighbors('b', searchRadius);
          rPixels.push(...rNeighbors);
          bPixels.push(...bNeighbors);
        } else {
          // Bayer: use specific neighbors based on row type
          const rNeighbors = collectNeighbors('r', searchRadius);
          const bNeighbors = collectNeighbors('b', searchRadius);
        const leftCh = getChannel(gx - 1, gy);
        const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
        if (isRedRow) {
            rPixels.push(...rNeighbors.filter(p => p.y === gy));
            bPixels.push(...bNeighbors.filter(p => p.x === gx));
        } else {
            rPixels.push(...rNeighbors.filter(p => p.x === gx));
            bPixels.push(...bNeighbors.filter(p => p.y === gy));
          }
        }
      }
    } else if (algorithm === 'lien_edge_based') {
      // Use collectNeighbors with radius 1
      if (centerCh === 'r') {
        const gNeighbors = collectNeighbors('g', 1);
        const bCorners = collectNeighbors('b', 1).filter(p => 
          (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
        );
        
        // Compute edge direction for green interpolation
        const gH = gNeighbors.filter(p => p.y === gy);
        const gV = gNeighbors.filter(p => p.x === gx);
        let diffH = 0, diffV = 0;
        if (gH.length >= 2) {
          const leftG = gH.find(p => p.x < gx);
          const rightG = gH.find(p => p.x > gx);
          if (leftG && rightG) {
            diffH = Math.abs(getVal(leftG.x, leftG.y) - getVal(rightG.x, rightG.y));
          }
        }
        if (gV.length >= 2) {
          const topG = gV.find(p => p.y < gy);
          const bottomG = gV.find(p => p.y > gy);
          if (topG && bottomG) {
            diffV = Math.abs(getVal(topG.x, topG.y) - getVal(bottomG.x, bottomG.y));
          }
        }
        
        // Mark dominant green neighbors: if diffH < diffV, horizontal is smoother, use horizontal neighbors
        // We interpolate along the smoother direction (perpendicular to the edge)
        const useHorizontal = diffH < diffV;
        gNeighbors.forEach(p => {
          const isHorizontal = (p.y === gy);
          p.dominant = useHorizontal ? isHorizontal : !isHorizontal;
          gPixels.push(p);
        });
        
        bPixels.push(...bCorners);
      } else if (centerCh === 'b') {
        const gNeighbors = collectNeighbors('g', 1);
        const rCorners = collectNeighbors('r', 1).filter(p => 
          (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
        );
        
        // Compute edge direction for green interpolation
        const gH = gNeighbors.filter(p => p.y === gy);
        const gV = gNeighbors.filter(p => p.x === gx);
        let diffH = 0, diffV = 0;
        if (gH.length >= 2) {
          const leftG = gH.find(p => p.x < gx);
          const rightG = gH.find(p => p.x > gx);
          if (leftG && rightG) {
            diffH = Math.abs(getVal(leftG.x, leftG.y) - getVal(rightG.x, rightG.y));
          }
        }
        if (gV.length >= 2) {
          const topG = gV.find(p => p.y < gy);
          const bottomG = gV.find(p => p.y > gy);
          if (topG && bottomG) {
            diffV = Math.abs(getVal(topG.x, topG.y) - getVal(bottomG.x, bottomG.y));
          }
        }
        
        // Mark dominant green neighbors: if diffH < diffV, horizontal is smoother, use horizontal neighbors
        // We interpolate along the smoother direction (perpendicular to the edge)
        const useHorizontal = diffH < diffV;
        gNeighbors.forEach(p => {
          const isHorizontal = (p.y === gy);
          p.dominant = useHorizontal ? isHorizontal : !isHorizontal;
          gPixels.push(p);
        });
        
        rPixels.push(...rCorners);
      } else { // Green
        // For green pixels, we interpolate RG and BG planes using edge detection
        // Collect all red and blue neighbors within radius 2 to include diagonals
        const rNeighbors = collectNeighbors('r', 2);
        const bNeighbors = collectNeighbors('b', 2);
        
        // Compute edge direction from RG plane (using red neighbors as proxy)
        // For Bayer: at green pixels, we need to estimate edge direction
        // Use immediate neighbors to determine edge direction
        const rH = rNeighbors.filter(p => p.y === gy);
        const rV = rNeighbors.filter(p => p.x === gx);
        const bH = bNeighbors.filter(p => p.y === gy);
        const bV = bNeighbors.filter(p => p.x === gx);
        
        // Estimate gradients from available neighbors
        let diffH = 0, diffV = 0;
        if (rH.length >= 2) {
          const leftR = rH.find(p => p.x < gx);
          const rightR = rH.find(p => p.x > gx);
          if (leftR && rightR) {
            diffH = Math.abs(getVal(leftR.x, leftR.y) - getVal(rightR.x, rightR.y));
          }
        }
        if (rV.length >= 2) {
          const topR = rV.find(p => p.y < gy);
          const bottomR = rV.find(p => p.y > gy);
          if (topR && bottomR) {
            diffV = Math.abs(getVal(topR.x, topR.y) - getVal(bottomR.x, bottomR.y));
          }
        }
        
        // If we can't determine from red neighbors, try blue
        if (diffH === 0 && diffV === 0) {
          if (bH.length >= 2) {
            const leftB = bH.find(p => p.x < gx);
            const rightB = bH.find(p => p.x > gx);
            if (leftB && rightB) {
              diffH = Math.abs(getVal(leftB.x, leftB.y) - getVal(rightB.x, rightB.y));
            }
          }
          if (bV.length >= 2) {
            const topB = bV.find(p => p.y < gy);
            const bottomB = bV.find(p => p.y > gy);
            if (topB && bottomB) {
              diffV = Math.abs(getVal(topB.x, topB.y) - getVal(bottomB.x, bottomB.y));
            }
          }
        }
        
        // Determine dominant direction: if diffH < diffV, horizontal is smoother, so edge is vertical (use H direction)
        const useHorizontal = diffH < diffV;
        const leftCh = getChannel(gx - 1, gy);
        const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
        
        // Mark dominant pixels based on edge direction and row type
        // For red: in red row, horizontal neighbors; in blue row, vertical neighbors
        // For blue: in red row, vertical neighbors; in blue row, horizontal neighbors
        rNeighbors.forEach(p => {
          const isHorizontal = (p.y === gy);
          const isVertical = (p.x === gx);
          let isDominant = false;
          
          if (isRedRow) {
            // Red row: R is horizontal, B is vertical
            isDominant = useHorizontal && isHorizontal;
          } else {
            // Blue row: R is vertical, B is horizontal
            isDominant = !useHorizontal && isVertical;
          }
          rPixels.push({...p, dominant: isDominant});
        });
        
        bNeighbors.forEach(p => {
          const isHorizontal = (p.y === gy);
          const isVertical = (p.x === gx);
          let isDominant = false;
          
          if (isRedRow) {
            // Red row: R is horizontal, B is vertical
            isDominant = !useHorizontal && isVertical;
          } else {
            // Blue row: R is vertical, B is horizontal
            isDominant = useHorizontal && isHorizontal;
          }
          bPixels.push({...p, dominant: isDominant});
        });
      }
    } else if (algorithm === 'wu_polynomial') {
      // Polynomial interpolation uses all neighbors within search radius (maxRadius = 5)
      const maxRadius = 5;
      if (centerCh === 'r') {
        const gNeighbors = collectNeighbors('g', maxRadius);
        const bNeighbors = collectNeighbors('b', maxRadius);
        gPixels.push(...gNeighbors);
        bPixels.push(...bNeighbors);
      } else if (centerCh === 'b') {
        const gNeighbors = collectNeighbors('g', maxRadius);
        const rNeighbors = collectNeighbors('r', maxRadius);
        gPixels.push(...gNeighbors);
        rPixels.push(...rNeighbors);
      } else { // Green
        const rNeighbors = collectNeighbors('r', maxRadius);
        const bNeighbors = collectNeighbors('b', maxRadius);
        rPixels.push(...rNeighbors);
        bPixels.push(...bNeighbors);
      }
    } else if (algorithm === 'kiku_residual') {
      // Residual interpolation uses bilinear for initial estimate (radius 1)
      if (centerCh === 'g') {
        const rNeighbors = collectNeighbors('r', 1);
        const bNeighbors = collectNeighbors('b', 1);
        const leftCh = getChannel(gx - 1, gy);
        const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
        if (isRedRow) {
          rPixels.push(...rNeighbors.filter(p => p.y === gy));
          bPixels.push(...bNeighbors.filter(p => p.x === gx));
        } else {
          rPixels.push(...rNeighbors.filter(p => p.x === gx));
          bPixels.push(...bNeighbors.filter(p => p.y === gy));
        }
      } else if (centerCh === 'r') {
        const gNeighbors = collectNeighbors('g', 1);
        const bCorners = collectNeighbors('b', 1).filter(p => 
          (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
        );
        gPixels.push(...gNeighbors);
        bPixels.push(...bCorners);
      } else { // Blue
        const gNeighbors = collectNeighbors('g', 1);
        const rCorners = collectNeighbors('r', 1).filter(p => 
          (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
        );
        gPixels.push(...gNeighbors);
        rPixels.push(...rCorners);
      }
    }
    
    return { rPixels, gPixels, bPixels };
  }, [globalCursorX, globalCursorY, algorithm, input]);

  // Draw Canvases
  useEffect(() => {
     if (inputCanvasRef.current) {
         inputCanvasRef.current.width = REGION_SIZE;
         inputCanvasRef.current.height = REGION_SIZE;
         const ctx = inputCanvasRef.current.getContext('2d');
         if (ctx) {
             ctx.imageSmoothingEnabled = false;
             ctx.putImageData(regionData.inputImageData, 0, 0);
         }
     }
     if (outputCanvasRef.current) {
         outputCanvasRef.current.width = REGION_SIZE;
         outputCanvasRef.current.height = REGION_SIZE;
         const ctx = outputCanvasRef.current.getContext('2d');
         if (ctx) {
             ctx.imageSmoothingEnabled = false;
             ctx.putImageData(regionData.outputImageData, 0, 0);
         }
     }
  }, [regionData]);

  // Draw Overlays
  useEffect(() => {
     const drawOverlay = (canvas: HTMLCanvasElement | null, isInput: boolean) => {
         if (!canvas || !inputCanvasRef.current) return;
         const rect = inputCanvasRef.current.getBoundingClientRect(); // Use input canvas for size ref
         const displayWidth = rect.width;
         const displayHeight = rect.height;
         
         canvas.width = displayWidth;
         canvas.height = displayHeight;
         const ctx = canvas.getContext('2d');
         if (!ctx) return;
         ctx.clearRect(0, 0, displayWidth, displayHeight);
         
         const scaleX = displayWidth / REGION_SIZE;
         const scaleY = displayHeight / REGION_SIZE;
         
         // Draw Grid
         ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
         ctx.lineWidth = 1;
         ctx.beginPath();
         for (let i = 1; i < REGION_SIZE; i++) {
             ctx.moveTo(i * scaleX, 0);
             ctx.lineTo(i * scaleX, displayHeight);
             ctx.moveTo(0, i * scaleY);
             ctx.lineTo(displayWidth, i * scaleY);
         }
         ctx.stroke();
         
         if (isInput) {
             // Draw center pixel highlight
             const centerX = localCursorX * scaleX + scaleX / 2;
             const centerY = localCursorY * scaleY + scaleY / 2;
             
             // Draw arrows to contributing pixels
             const drawArrow = (fromGlobalX: number, fromGlobalY: number, toGlobalX: number, toGlobalY: number, color: string, lineWidth: number = 2) => {
                 // Convert to local coordinates
                 const fromLocalX = fromGlobalX - regionOriginX;
                 const fromLocalY = fromGlobalY - regionOriginY;
                 const toLocalX = toGlobalX - regionOriginX;
                 const toLocalY = toGlobalY - regionOriginY;
                 
                 // Only draw if both pixels are within region bounds
                 if (fromLocalX < 0 || fromLocalX >= REGION_SIZE || 
                     fromLocalY < 0 || fromLocalY >= REGION_SIZE ||
                     toLocalX < 0 || toLocalX >= REGION_SIZE || 
                     toLocalY < 0 || toLocalY >= REGION_SIZE) {
                     return;
                 }
                 
                 const fx = fromLocalX * scaleX + scaleX / 2;
                 const fy = fromLocalY * scaleY + scaleY / 2;
                 const tx = toLocalX * scaleX + scaleX / 2;
                 const ty = toLocalY * scaleY + scaleY / 2;
                 
                const angle = Math.atan2(ty - fy, tx - fx);
                const arrowLength = Math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2);
                // Make arrowhead size proportional to pixel size, but cap it
                const arrowHeadLength = Math.min(Math.max(scaleX * 0.4, 4), arrowLength * 0.4);
                const arrowHeadAngle = Math.PI / 6;
                
                // Calculate where the arrowhead base should be (at the edge of target pixel)
                const arrowHeadBaseX = tx - arrowHeadLength * Math.cos(angle);
                const arrowHeadBaseY = ty - arrowHeadLength * Math.sin(angle);
                
                // Draw arrow line - extends all the way to the arrowhead base
                ctx.strokeStyle = color;
                ctx.lineWidth = lineWidth;
                ctx.beginPath();
                ctx.moveTo(fx, fy);
                ctx.lineTo(arrowHeadBaseX, arrowHeadBaseY);
                ctx.stroke();
                
                // Draw arrowhead at the target pixel center
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.moveTo(tx, ty);
                ctx.lineTo(
                    tx - arrowHeadLength * Math.cos(angle - arrowHeadAngle),
                    ty - arrowHeadLength * Math.sin(angle - arrowHeadAngle)
                );
                ctx.lineTo(
                    tx - arrowHeadLength * Math.cos(angle + arrowHeadAngle),
                    ty - arrowHeadLength * Math.sin(angle + arrowHeadAngle)
                );
                ctx.closePath();
                ctx.fill();
                
                // Draw a highlight circle for dominant pixels
                if (lineWidth > 2) {
                    ctx.strokeStyle = '#ffff00'; // Yellow highlight
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(tx, ty, Math.max(scaleX * 0.3, 3), 0, Math.PI * 2);
                    ctx.stroke();
                }
             };
             
             // Draw red arrows (non-dominant first, then dominant on top)
             contributingPixels.rPixels.filter(p => !p.dominant).forEach(pixel => {
                 if (pixel.x !== globalCursorX || pixel.y !== globalCursorY) {
                     drawArrow(globalCursorX, globalCursorY, pixel.x, pixel.y, '#ff0000', 2);
                 }
             });
             contributingPixels.rPixels.filter(p => p.dominant === true).forEach(pixel => {
                 if (pixel.x !== globalCursorX || pixel.y !== globalCursorY) {
                     drawArrow(globalCursorX, globalCursorY, pixel.x, pixel.y, '#ff0000', 4);
                 }
             });
             
             // Draw green arrows
             contributingPixels.gPixels.filter(p => !p.dominant).forEach(pixel => {
                 if (pixel.x !== globalCursorX || pixel.y !== globalCursorY) {
                     drawArrow(globalCursorX, globalCursorY, pixel.x, pixel.y, '#00ff00', 2);
                 }
             });
             contributingPixels.gPixels.filter(p => p.dominant === true).forEach(pixel => {
                 if (pixel.x !== globalCursorX || pixel.y !== globalCursorY) {
                     drawArrow(globalCursorX, globalCursorY, pixel.x, pixel.y, '#00ff00', 4);
                 }
             });
             
             // Draw blue arrows (non-dominant first, then dominant on top)
             contributingPixels.bPixels.filter(p => !p.dominant).forEach(pixel => {
                 if (pixel.x !== globalCursorX || pixel.y !== globalCursorY) {
                     drawArrow(globalCursorX, globalCursorY, pixel.x, pixel.y, '#0000ff', 2);
                 }
             });
             contributingPixels.bPixels.filter(p => p.dominant === true).forEach(pixel => {
                 if (pixel.x !== globalCursorX || pixel.y !== globalCursorY) {
                     drawArrow(globalCursorX, globalCursorY, pixel.x, pixel.y, '#0000ff', 4);
                 }
             });
             
             // Draw center pixel highlight
             ctx.strokeStyle = "#ffffff";
             ctx.lineWidth = 2;
             ctx.strokeRect(
                 localCursorX * scaleX,
                 localCursorY * scaleY,
                 scaleX,
                 scaleY
             );
         } else {
             // Output overlay: just show selected pixel
             const kx = localCursorX * scaleX;
             const ky = localCursorY * scaleY;
             ctx.strokeStyle = "#ff0000";
             ctx.lineWidth = 2;
             ctx.setLineDash([]);
             ctx.strokeRect(kx, ky, scaleX, scaleY);
         }
     };
     
     // Need to wait for layout?
     requestAnimationFrame(() => {
        drawOverlay(inputOverlayRef.current, true);
        drawOverlay(outputOverlayRef.current, false);
     });
     
  }, [localCursorX, localCursorY, contributingPixels, globalCursorX, globalCursorY, regionOriginX, regionOriginY]);

  // Mouse Handlers
  const getCoords = (e: React.MouseEvent) => {
      const canvas = inputCanvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const x = Math.floor((e.clientX - rect.left) / (rect.width / REGION_SIZE));
      const y = Math.floor((e.clientY - rect.top) / (rect.height / REGION_SIZE));
      return { x: Math.max(0, Math.min(REGION_SIZE-1, x)), y: Math.max(0, Math.min(REGION_SIZE-1, y)) };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
      const c = getCoords(e);
      if (c) {
          setLocalCursorX(c.x);
          setLocalCursorY(c.y);
          setIsDragging(true);
      }
  };
  
  const handleMouseMove = (e: React.MouseEvent) => {
      if (!isDragging) return;
      const c = getCoords(e);
      if (c) {
          setLocalCursorX(c.x);
          setLocalCursorY(c.y);
      }
  };

  // Compute hyperparameters display
  const hyperparameters = useMemo(() => {
    const hyperparams: Array<{ label: string; value: string | number }> = [];
    
    if (algorithm === 'niu_edge_sensing') {
      const threshold = params?.niuLogisticThreshold ?? 0.1;
      hyperparams.push({ label: 'Edge Detection Threshold (θ)', value: threshold });
      const k = params?.niuLogisticSteepness ?? (20.0 / Math.max(0.01, threshold));
      hyperparams.push({ label: 'Logistic Steepness (k)', value: k.toFixed(2) });
    } else if (algorithm === 'wu_polynomial') {
      const degree = params?.wuPolynomialDegree ?? 2;
      hyperparams.push({ label: 'Polynomial Degree', value: degree });
    } else if (algorithm === 'kiku_residual') {
      const iterations = params?.kikuResidualIterations ?? 1;
      hyperparams.push({ label: 'Residual Iterations', value: iterations });
    }
    
    return hyperparams;
  }, [algorithm, params]);

  // Compute detailed calculation breakdown for the selected pixel
  const calculationBreakdown = useMemo(() => {
    const gx = globalCursorX;
    const gy = globalCursorY;
    const { width, height, cfaPatternMeta, cfaPattern } = input;
    const getChannel: (x: number, y: number) => 'r' | 'g' | 'b' = cfaPattern === 'bayer' 
      ? getBayerKernel(cfaPatternMeta.layout) 
      : cfaPattern === 'xtrans'
      ? getXTransKernel()
      : () => 'g' as const;
    const getVal = (x: number, y: number) => {
      let sx = x; 
      if (sx < 0) sx = -sx; 
      else if (sx >= width) sx = 2*width - 2 - sx;
      let sy = y; 
      if (sy < 0) sy = -sy; 
      else if (sy >= height) sy = 2*height - 2 - sy;
      sx = Math.max(0, Math.min(width - 1, sx));
      sy = Math.max(0, Math.min(height - 1, sy));
      return input.cfaData[sy * width + sx];
    };
    
    const centerCh = getChannel(gx, gy);
    const centerVal = getVal(gx, gy);
    const breakdown: Array<{ step: string; description: string; formula?: string; result: string }> = [];
    
    // Convert to 0-255 range for display
    const to255 = (val: number) => Math.round(val * 255);
    
    const centerRGB = centerCh === 'r' 
      ? `(${to255(centerVal)}, 0, 0)`
      : centerCh === 'g'
      ? `(0, ${to255(centerVal)}, 0)`
      : `(0, 0, ${to255(centerVal)})`;
    breakdown.push({
      step: '1. Raw Sensor Value',
      description: `Pixel at (${gx}, ${gy}) has ${centerCh.toUpperCase()} channel`,
      formula: `I_{sensor}(${gx}, ${gy}) = ${centerCh.toUpperCase()}`,
      result: centerRGB
    });
    
    // Helper function for collectNeighbors
    const collectNeighbors = (targetColor: 'r' | 'g' | 'b', maxRadius: number = 10) => {
        const result: { values: number[]; positions: Array<{x: number, y: number, val: number}> } = { values: [], positions: [] };
        for (let dy = -maxRadius; dy <= maxRadius; dy++) {
            for (let dx = -maxRadius; dx <= maxRadius; dx++) {
                if (dx === 0 && dy === 0) continue;
                const px = gx + dx;
                const py = gy + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    const ch = getChannel(px, py);
                    if (ch === targetColor) {
                        const val = getVal(px, py);
                        result.values.push(val);
                        result.positions.push({x: px, y: py, val});
                    }
                }
            }
        }
        return result;
    };
    
    // Compute RGB values using the same logic as regionData
    let r = 0, g = 0, b = 0;
    // Store neighbor data for breakdown display
    let neighborData: {
        rNeighbors?: Array<{x: number, y: number, val: number}>;
        gNeighbors?: Array<{x: number, y: number, val: number}>;
        bNeighbors?: Array<{x: number, y: number, val: number}>;
    } = {};
    // Extra detail for the residual-based method so the breakdown can show both the baseline and residual corrections
    let kikuResidualDetail: {
      initialR: number;
      initialG: number;
      initialB: number;
      residualSamplesR: Array<{x: number, y: number, val: number, residual: number}>;
      residualSamplesG: Array<{x: number, y: number, val: number, residual: number}>;
      residualSamplesB: Array<{x: number, y: number, val: number, residual: number}>;
      residualApplied: { r: number; g: number; b: number };
      iterations: number;
    } | null = null;
    
    if (algorithm === 'nearest') {
        // Find nearest neighbors using expanding search
        if (centerCh === 'r') {
            r = centerVal;
            let gFound: {x: number, y: number, val: number} | null = null;
            let bFound: {x: number, y: number, val: number} | null = null;
            for (let d = 1; d <= 10; d++) {
                for (let dy = -d; dy <= d; dy++) {
                    for (let dx = -d; dx <= d; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist > d - 1 && dist <= d) {
                            const px = gx + dx;
                            const py = gy + dy;
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const nch = getChannel(px, py);
                                const nval = getVal(px, py);
                                if (nch === 'g' && !gFound) {
                                    gFound = {x: px, y: py, val: nval};
                                    g = nval;
                                }
                                if (nch === 'b' && !bFound) {
                                    bFound = {x: px, y: py, val: nval};
                                    b = nval;
                                }
                            }
                        }
                    }
                }
                if (gFound && bFound) break;
            }
            if (gFound) neighborData.gNeighbors = [gFound];
            if (bFound) neighborData.bNeighbors = [bFound];
        } else if (centerCh === 'b') {
            b = centerVal;
            let gFound: {x: number, y: number, val: number} | null = null;
            let rFound: {x: number, y: number, val: number} | null = null;
            for (let d = 1; d <= 10; d++) {
                for (let dy = -d; dy <= d; dy++) {
                    for (let dx = -d; dx <= d; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist > d - 1 && dist <= d) {
                            const px = gx + dx;
                            const py = gy + dy;
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const nch = getChannel(px, py);
                                const nval = getVal(px, py);
                                if (nch === 'g' && !gFound) {
                                    gFound = {x: px, y: py, val: nval};
                                    g = nval;
                                }
                                if (nch === 'r' && !rFound) {
                                    rFound = {x: px, y: py, val: nval};
                                    r = nval;
                                }
                            }
                        }
                    }
                }
                if (gFound && rFound) break;
            }
            if (gFound) neighborData.gNeighbors = [gFound];
            if (rFound) neighborData.rNeighbors = [rFound];
        } else {
            g = centerVal;
            let rFound: {x: number, y: number, val: number} | null = null;
            let bFound: {x: number, y: number, val: number} | null = null;
            for (let d = 1; d <= 10; d++) {
                for (let dy = -d; dy <= d; dy++) {
                    for (let dx = -d; dx <= d; dx++) {
                        if (dx === 0 && dy === 0) continue;
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist > d - 1 && dist <= d) {
                            const px = gx + dx;
                            const py = gy + dy;
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const nch = getChannel(px, py);
                                const nval = getVal(px, py);
                                if (nch === 'r' && !rFound) {
                                    rFound = {x: px, y: py, val: nval};
                                    r = nval;
                                }
                                if (nch === 'b' && !bFound) {
                                    bFound = {x: px, y: py, val: nval};
                                    b = nval;
                                }
                            }
                        }
                    }
                }
                if (rFound && bFound) break;
            }
            if (rFound) neighborData.rNeighbors = [rFound];
            if (bFound) neighborData.bNeighbors = [bFound];
        }
    } else if (algorithm === 'bilinear') {
        // Use collectNeighbors for actual implementation matching (radius 1)
        if (centerCh === 'g') {
            g = centerVal;
            const rNeighbors = collectNeighbors('r', 1);
            const bNeighbors = collectNeighbors('b', 1);
            neighborData.rNeighbors = rNeighbors.positions;
            neighborData.bNeighbors = bNeighbors.positions;
            r = rNeighbors.values.length > 0 ? rNeighbors.values.reduce((a, b) => a + b, 0) / rNeighbors.values.length : 0;
            b = bNeighbors.values.length > 0 ? bNeighbors.values.reduce((a, b) => a + b, 0) / bNeighbors.values.length : 0;
        } else if (centerCh === 'r') {
            r = centerVal;
            const gNeighbors = collectNeighbors('g', 1);
            const bNeighbors = collectNeighbors('b', 1);
            neighborData.gNeighbors = gNeighbors.positions;
            neighborData.bNeighbors = bNeighbors.positions;
            g = gNeighbors.values.length > 0 ? gNeighbors.values.reduce((a, b) => a + b, 0) / gNeighbors.values.length : 0;
            b = bNeighbors.values.length > 0 ? bNeighbors.values.reduce((a, b) => a + b, 0) / bNeighbors.values.length : 0;
        } else {
            b = centerVal;
            const gNeighbors = collectNeighbors('g', 1);
            const rNeighbors = collectNeighbors('r', 1);
            neighborData.gNeighbors = gNeighbors.positions;
            neighborData.rNeighbors = rNeighbors.positions;
            g = gNeighbors.values.length > 0 ? gNeighbors.values.reduce((a, b) => a + b, 0) / gNeighbors.values.length : 0;
            r = rNeighbors.values.length > 0 ? rNeighbors.values.reduce((a, b) => a + b, 0) / rNeighbors.values.length : 0;
        }
    } else if (algorithm === 'niu_edge_sensing') {
        // X-Trans uses radius 2 (5x5), Bayer uses radius 1 (3x3)
        const searchRadius = cfaPattern === 'xtrans' ? 2 : 1;
        let greenInterp = 0;
        if (centerCh === 'g') {
            greenInterp = centerVal;
        } else {
            const threshold = params?.niuLogisticThreshold ?? 0.1;
            const steepness = params?.niuLogisticSteepness;
            const vars = computeDirectionalVariations(input.cfaData, width, height, gx, gy, getChannel, getVal);
            const wH = logisticFunction(vars.horizontal, threshold, steepness);
            const wV = logisticFunction(vars.vertical, threshold, steepness);
            const sumW = wH + wV;
            const nH = (sumW > 0) ? (1.0 - wH / sumW) : 0.5;
            const nV = (sumW > 0) ? (1.0 - wV / sumW) : 0.5;
            
            if (cfaPattern === 'xtrans') {
                // X-Trans: collect horizontal and vertical green neighbors separately
                // Then use edge-aware weighting (nH, nV)
                const greenH = collectNeighbors('g', searchRadius).positions.filter(p => p.y === gy);
                const greenV = collectNeighbors('g', searchRadius).positions.filter(p => p.x === gx);
                neighborData.gNeighbors = [...greenH, ...greenV];
                const gH = greenH.length > 0 ? greenH.reduce((sum, p) => sum + p.val, 0) / greenH.length : 0;
                const gV = greenV.length > 0 ? greenV.reduce((sum, p) => sum + p.val, 0) / greenV.length : 0;
                greenInterp = (gH * nH + gV * nV) / (nH + nV);
            } else {
                // Bayer: use horizontal/vertical green neighbors only
                const greenH = collectNeighbors('g', searchRadius).positions.filter(p => p.y === gy);
                const greenV = collectNeighbors('g', searchRadius).positions.filter(p => p.x === gx);
                neighborData.gNeighbors = [...greenH, ...greenV];
                const gH = greenH.length > 0 ? greenH.reduce((sum, p) => sum + p.val, 0) / greenH.length : 0;
                const gV = greenV.length > 0 ? greenV.reduce((sum, p) => sum + p.val, 0) / greenV.length : 0;
            greenInterp = (gH * nH + gV * nV) / (nH + nV);
            }
        }
        if (centerCh === 'r') {
            r = centerVal;
            g = greenInterp;
            if (cfaPattern === 'xtrans') {
                // X-Trans: collect all blue neighbors in 5x5 neighborhood
                const bAll = collectNeighbors('b', searchRadius).positions;
                neighborData.bNeighbors = bAll;
                const bSum = bAll.reduce((sum, p) => sum + p.val, 0);
                b = bAll.length > 0 ? bSum / bAll.length : g;
            } else {
                // Bayer: use corner blue neighbors only
                const bCorners = collectNeighbors('b', searchRadius).positions.filter(p => 
                    (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
                );
                neighborData.bNeighbors = bCorners;
                const bMinusG = bCorners.map(p => p.val - greenInterp);
                const avgBMinusG = bMinusG.length > 0 ? bMinusG.reduce((a, b) => a + b, 0) / bMinusG.length : 0;
            b = g + avgBMinusG;
            }
        } else if (centerCh === 'b') {
            b = centerVal;
            g = greenInterp;
            if (cfaPattern === 'xtrans') {
                // X-Trans: collect all red neighbors in 5x5 neighborhood
                const rAll = collectNeighbors('r', searchRadius).positions;
                neighborData.rNeighbors = rAll;
                const rSum = rAll.reduce((sum, p) => sum + p.val, 0);
                r = rAll.length > 0 ? rSum / rAll.length : g;
            } else {
                // Bayer: use corner red neighbors only
                const rCorners = collectNeighbors('r', searchRadius).positions.filter(p => 
                    (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
                );
                neighborData.rNeighbors = rCorners;
                const rMinusG = rCorners.map(p => p.val - greenInterp);
                const avgRMinusG = rMinusG.length > 0 ? rMinusG.reduce((a, b) => a + b, 0) / rMinusG.length : 0;
            r = g + avgRMinusG;
            }
        } else {
            g = centerVal;
            if (cfaPattern === 'xtrans') {
                // X-Trans: collect all R/B neighbors in 5x5 neighborhood
                const rAll = collectNeighbors('r', searchRadius).positions;
                const bAll = collectNeighbors('b', searchRadius).positions;
                neighborData.rNeighbors = rAll;
                neighborData.bNeighbors = bAll;
                const rSum = rAll.reduce((sum, p) => sum + p.val, 0);
                const bSum = bAll.reduce((sum, p) => sum + p.val, 0);
                r = rAll.length > 0 ? rSum / rAll.length : 0;
                b = bAll.length > 0 ? bSum / bAll.length : 0;
            } else {
                // Bayer: use specific neighbors based on row type
                const rNeighbors = collectNeighbors('r', searchRadius).positions;
                const bNeighbors = collectNeighbors('b', searchRadius).positions;
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            if (isRedRow) {
                    neighborData.rNeighbors = rNeighbors.filter(p => p.y === gy);
                    neighborData.bNeighbors = bNeighbors.filter(p => p.x === gx);
                    const rVals = neighborData.rNeighbors.map(p => p.val);
                    const bVals = neighborData.bNeighbors.map(p => p.val);
                    r = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
                    b = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
            } else {
                    neighborData.rNeighbors = rNeighbors.filter(p => p.x === gx);
                    neighborData.bNeighbors = bNeighbors.filter(p => p.y === gy);
                    const rVals = neighborData.rNeighbors.map(p => p.val);
                    const bVals = neighborData.bNeighbors.map(p => p.val);
                    r = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
                    b = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
                }
            }
        }
    } else if (algorithm === 'lien_edge_based') {
        // X-Trans uses radius 2 (5x5) for R/B when center is G, radius 1 for green when center is R/B
        const searchRadiusRB = cfaPattern === 'xtrans' ? 2 : 1;
        const searchRadiusG = 1; // Always use immediate neighbors for green interpolation
        if (centerCh === 'g') {
            g = centerVal;
            const diffH = Math.abs(getVal(gx - 1, gy) - getVal(gx + 1, gy));
            const diffV = Math.abs(getVal(gx, gy - 1) - getVal(gx, gy + 1));
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            const rNeighbors = collectNeighbors('r', searchRadiusRB).positions;
            const bNeighbors = collectNeighbors('b', searchRadiusRB).positions;
            if (isRedRow) {
                if (diffH < diffV) {
                    neighborData.rNeighbors = rNeighbors.filter(p => p.y === gy);
                    neighborData.bNeighbors = bNeighbors.filter(p => p.x === gx);
                    const rVals = neighborData.rNeighbors.map(p => p.val);
                    const bVals = neighborData.bNeighbors.map(p => p.val);
                    r = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
                    b = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
                } else {
                    neighborData.rNeighbors = rNeighbors.filter(p => p.x === gx);
                    neighborData.bNeighbors = bNeighbors.filter(p => p.y === gy);
                    const rVals = neighborData.rNeighbors.map(p => p.val);
                    const bVals = neighborData.bNeighbors.map(p => p.val);
                    r = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
                    b = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
                }
            } else {
                if (diffH < diffV) {
                    neighborData.rNeighbors = rNeighbors.filter(p => p.x === gx);
                    neighborData.bNeighbors = bNeighbors.filter(p => p.y === gy);
                    const rVals = neighborData.rNeighbors.map(p => p.val);
                    const bVals = neighborData.bNeighbors.map(p => p.val);
                    r = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
                    b = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
                } else {
                    neighborData.rNeighbors = rNeighbors.filter(p => p.y === gy);
                    neighborData.bNeighbors = bNeighbors.filter(p => p.x === gx);
                    const rVals = neighborData.rNeighbors.map(p => p.val);
                    const bVals = neighborData.bNeighbors.map(p => p.val);
                    r = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
                    b = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
                }
            }
        } else if (centerCh === 'r') {
            r = centerVal;
            const diffH = Math.abs(getVal(gx - 1, gy) - getVal(gx + 1, gy));
            const diffV = Math.abs(getVal(gx, gy - 1) - getVal(gx, gy + 1));
            const gNeighbors = collectNeighbors('g', searchRadiusG).positions;
            if (cfaPattern === 'xtrans') {
                // X-Trans: collect all blue neighbors in 5x5 neighborhood
                const bAll = collectNeighbors('b', searchRadiusRB).positions;
                neighborData.bNeighbors = bAll;
                // For green, use edge-aware selection from immediate neighbors
            if (diffH < diffV) {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.y === gy);
            } else {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.x === gx);
                }
                const gVals = neighborData.gNeighbors.map(p => p.val);
                g = gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : 0;
                const bSum = bAll.reduce((sum, p) => sum + p.val, 0);
                b = bAll.length > 0 ? bSum / bAll.length : g;
            } else {
                // Bayer: use corner blue neighbors only
                const bCorners = collectNeighbors('b', searchRadiusRB).positions.filter(p => 
                    (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
                );
                neighborData.bNeighbors = bCorners;
                if (diffH < diffV) {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.y === gy);
                } else {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.x === gx);
                }
                const gVals = neighborData.gNeighbors.map(p => p.val);
                g = gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : 0;
                const avgB = bCorners.length > 0 ? bCorners.reduce((sum, p) => sum + p.val, 0) / bCorners.length : 0;
                const avgG = gNeighbors.length > 0 ? gNeighbors.reduce((sum, p) => sum + p.val, 0) / gNeighbors.length : 0;
            b = avgB + (g - avgG);
            }
        } else {
            b = centerVal;
            const diffH = Math.abs(getVal(gx - 1, gy) - getVal(gx + 1, gy));
            const diffV = Math.abs(getVal(gx, gy - 1) - getVal(gx, gy + 1));
            const gNeighbors = collectNeighbors('g', searchRadiusG).positions;
            if (cfaPattern === 'xtrans') {
                // X-Trans: collect all red neighbors in 5x5 neighborhood
                const rAll = collectNeighbors('r', searchRadiusRB).positions;
                neighborData.rNeighbors = rAll;
                // For green, use edge-aware selection from immediate neighbors
            if (diffH < diffV) {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.y === gy);
            } else {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.x === gx);
                }
                const gVals = neighborData.gNeighbors.map(p => p.val);
                g = gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : 0;
                const rSum = rAll.reduce((sum, p) => sum + p.val, 0);
                r = rAll.length > 0 ? rSum / rAll.length : g;
            } else {
                // Bayer: use corner red neighbors only
                const rCorners = collectNeighbors('r', searchRadiusRB).positions.filter(p => 
                    (p.x === gx - 1 || p.x === gx + 1) && (p.y === gy - 1 || p.y === gy + 1)
                );
                neighborData.rNeighbors = rCorners;
                if (diffH < diffV) {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.y === gy);
                } else {
                    neighborData.gNeighbors = gNeighbors.filter(p => p.x === gx);
                }
                const gVals = neighborData.gNeighbors.map(p => p.val);
                g = gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : 0;
                const avgR = rCorners.length > 0 ? rCorners.reduce((sum, p) => sum + p.val, 0) / rCorners.length : 0;
                const avgG = gNeighbors.length > 0 ? gNeighbors.reduce((sum, p) => sum + p.val, 0) / gNeighbors.length : 0;
            r = avgR + (g - avgG);
            }
        }
    } else if (algorithm === 'wu_polynomial') {
        const degree = params?.wuPolynomialDegree ?? 2;
        let greenInterp = 0;
        if (centerCh === 'g') {
            greenInterp = centerVal;
        } else {
            const gNeighbors = collectNeighbors('g', 5);
            neighborData.gNeighbors = gNeighbors.positions;
            if (gNeighbors.values.length > 0) {
                // Calculate distances for polynomial interpolation
                const distances = gNeighbors.positions.map(p => 
                    Math.sqrt((p.x - gx) ** 2 + (p.y - gy) ** 2)
                );
                greenInterp = polynomialInterpolate(gNeighbors.values, distances, degree);
            } else {
                greenInterp = centerVal;
            }
        }
        if (centerCh === 'r') {
            r = centerVal;
            g = greenInterp;
            const bNeighbors = collectNeighbors('b', 5);
            neighborData.bNeighbors = bNeighbors.positions;
            if (bNeighbors.values.length > 0) {
                const distances = bNeighbors.positions.map(p => 
                    Math.sqrt((p.x - gx) ** 2 + (p.y - gy) ** 2)
                );
                b = polynomialInterpolate(bNeighbors.values, distances, degree);
            } else {
                b = g;
            }
        } else if (centerCh === 'b') {
            b = centerVal;
            g = greenInterp;
            const rNeighbors = collectNeighbors('r', 5);
            neighborData.rNeighbors = rNeighbors.positions;
            if (rNeighbors.values.length > 0) {
                const distances = rNeighbors.positions.map(p => 
                    Math.sqrt((p.x - gx) ** 2 + (p.y - gy) ** 2)
                );
                r = polynomialInterpolate(rNeighbors.values, distances, degree);
            } else {
                r = g;
            }
        } else {
            g = centerVal;
            const rNeighbors = collectNeighbors('r', 5);
            const bNeighbors = collectNeighbors('b', 5);
            neighborData.rNeighbors = rNeighbors.positions;
            neighborData.bNeighbors = bNeighbors.positions;
            if (rNeighbors.values.length > 0) {
                const distances = rNeighbors.positions.map(p => 
                    Math.sqrt((p.x - gx) ** 2 + (p.y - gy) ** 2)
                );
                r = polynomialInterpolate(rNeighbors.values, distances, degree);
            }
            if (bNeighbors.values.length > 0) {
                const distances = bNeighbors.positions.map(p => 
                    Math.sqrt((p.x - gx) ** 2 + (p.y - gy) ** 2)
                );
                b = polynomialInterpolate(bNeighbors.values, distances, degree);
            }
        }
    } else if (algorithm === 'kiku_residual') {
        const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

        // Bilinear estimate helper (mirrors the region visualizer logic above)
        const getInitialEstimate = (px: number, py: number, targetCh: 'r' | 'g' | 'b'): number => {
            // Always compute a bilinear-style estimate, even if the pixel already has that channel.
            // This avoids zero residuals caused by short-circuiting to the observed value.
            if (targetCh === 'g') {
                // Cross average of green neighbors
                const gVals = [
                    getVal(px - 1, py), getVal(px + 1, py),
                    getVal(px, py - 1), getVal(px, py + 1)
                ].filter((v) => v !== undefined);
                return gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : getVal(px, py);
            }
            if (targetCh === 'r') {
                // Average red corners
                const rVals = [
                    getVal(px - 1, py - 1), getVal(px + 1, py - 1),
                    getVal(px - 1, py + 1), getVal(px + 1, py + 1)
                ].filter((v) => v !== undefined);
                return rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : getVal(px, py);
            }
            // targetCh === 'b'
            const bVals = [
                getVal(px - 1, py - 1), getVal(px + 1, py - 1),
                getVal(px - 1, py + 1), getVal(px + 1, py + 1)
            ].filter((v) => v !== undefined);
            return bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : getVal(px, py);
        };

        // Baseline (bilinear) estimate at the center pixel
        let initialR = 0, initialG = 0, initialB = 0;
        if (centerCh === 'g') {
            initialG = centerVal;
            const leftCh = getChannel(gx - 1, gy);
            const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
            if (isRedRow) {
                initialR = (getVal(gx-1, gy) + getVal(gx+1, gy)) / 2;
                initialB = (getVal(gx, gy-1) + getVal(gx, gy+1)) / 2;
            } else {
                initialR = (getVal(gx, gy-1) + getVal(gx, gy+1)) / 2;
                initialB = (getVal(gx-1, gy) + getVal(gx+1, gy)) / 2;
            }
        } else if (centerCh === 'r') {
            initialR = centerVal;
            initialG = (getVal(gx-1, gy) + getVal(gx+1, gy) + getVal(gx, gy-1) + getVal(gx, gy+1)) / 4;
            initialB = (getVal(gx-1, gy-1) + getVal(gx+1, gy-1) + getVal(gx-1, gy+1) + getVal(gx+1, gy+1)) / 4;
        } else {
            initialB = centerVal;
            initialG = (getVal(gx-1, gy) + getVal(gx+1, gy) + getVal(gx, gy-1) + getVal(gx, gy+1)) / 4;
            initialR = (getVal(gx-1, gy-1) + getVal(gx+1, gy-1) + getVal(gx-1, gy+1) + getVal(gx+1, gy+1)) / 4;
        }

        // Collect residuals from surrounding measured pixels of each channel
        const collectResidualNeighborsWithPos = (
          targetColor: 'r' | 'g' | 'b',
          maxRadius: number = 5
        ): Array<{x: number, y: number, val: number, residual: number}> => {
          const results: Array<{x: number, y: number, val: number, residual: number}> = [];
          for (let dy = -maxRadius; dy <= maxRadius; dy++) {
            for (let dx = -maxRadius; dx <= maxRadius; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = gx + dx;
              const ny = gy + dy;
              if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
              const nch = getChannel(nx, ny);
              if (nch === targetColor) {
                const observed = getVal(nx, ny);
                const estimate = getInitialEstimate(nx, ny, targetColor);
                results.push({ x: nx, y: ny, val: observed, residual: observed - estimate });
              }
            }
          }
          return results;
        };

        const residualSamplesR = collectResidualNeighborsWithPos('r', 5);
        const residualSamplesG = collectResidualNeighborsWithPos('g', 5);
        const residualSamplesB = collectResidualNeighborsWithPos('b', 5);

        const meanResidual = (samples: Array<{ residual: number }>) =>
          samples.length > 0 ? samples.reduce((sum, s) => sum + s.residual, 0) / samples.length : 0;

        const residualR = centerCh === 'r' ? (centerVal - initialR) : meanResidual(residualSamplesR);
        const residualG = centerCh === 'g' ? (centerVal - initialG) : meanResidual(residualSamplesG);
        const residualB = centerCh === 'b' ? (centerVal - initialB) : meanResidual(residualSamplesB);

        r = clamp01(initialR + residualR);
        g = clamp01(initialG + residualG);
        b = clamp01(initialB + residualB);

        // Track neighbors for overlays and breakdown
        neighborData = {
          rNeighbors: residualSamplesR.map(({ x, y, val }) => ({ x, y, val })),
          gNeighbors: residualSamplesG.map(({ x, y, val }) => ({ x, y, val })),
          bNeighbors: residualSamplesB.map(({ x, y, val }) => ({ x, y, val })),
        };

        kikuResidualDetail = {
          initialR,
          initialG,
          initialB,
          residualSamplesR,
          residualSamplesG,
          residualSamplesB,
          residualApplied: { r: residualR, g: residualG, b: residualB },
          iterations: params?.kikuResidualIterations ?? 1,
        };
    }
    
    if (algorithm === 'nearest') {
      if (centerCh === 'r') {
        if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0) {
          const gVals = neighborData.gNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '2. Find Nearest Green',
            description: `Nearest green pixel`,
            formula: `Ĝ = G(nearest)`,
            result: `${gVals.map(v => `(0, ${v}, 0)`).join(', ')} → ${to255(g)}`
          });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          const bVals = neighborData.bNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '3. Find Nearest Blue',
            description: `Nearest blue pixel`,
            formula: `B̂ = B(nearest)`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
          });
        }
      } else if (centerCh === 'b') {
        if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0) {
          const gVals = neighborData.gNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '2. Find Nearest Green',
            description: `Nearest green pixel`,
            formula: `Ĝ = G(nearest)`,
            result: `${gVals.map(v => `(0, ${v}, 0)`).join(', ')} → ${to255(g)}`
          });
        }
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          const rVals = neighborData.rNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '3. Find Nearest Red',
            description: `Nearest red pixel`,
            formula: `R̂ = R(nearest)`,
            result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
          });
        }
        } else {
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          const rVals = neighborData.rNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '2. Find Nearest Red',
            description: `Nearest red pixel`,
            formula: `R̂ = R(nearest)`,
            result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
          });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          const bVals = neighborData.bNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '3. Find Nearest Blue',
            description: `Nearest blue pixel`,
            formula: `B̂ = B(nearest)`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
          });
        }
      }
    } else if (algorithm === 'bilinear') {
      if (centerCh === 'g') {
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          const rVals = neighborData.rNeighbors.map(n => to255(n.val));
          const rSum = rVals.reduce((a, b) => a + b, 0);
          breakdown.push({
            step: '2. Interpolate Red',
            description: `Average of ${neighborData.rNeighbors.length} red neighbor${neighborData.rNeighbors.length > 1 ? 's' : ''}`,
            formula: `R̂ = (${rVals.join(' + ')}) / ${neighborData.rNeighbors.length}`,
            result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
          });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          const bVals = neighborData.bNeighbors.map(n => to255(n.val));
          const bSum = bVals.reduce((a, b) => a + b, 0);
          breakdown.push({
            step: '3. Interpolate Blue',
            description: `Average of ${neighborData.bNeighbors.length} blue neighbor${neighborData.bNeighbors.length > 1 ? 's' : ''}`,
            formula: `B̂ = (${bVals.join(' + ')}) / ${neighborData.bNeighbors.length}`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
          });
        }
      } else if (centerCh === 'r') {
        if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0) {
          const gVals = neighborData.gNeighbors.map(n => to255(n.val));
        breakdown.push({
            step: '2. Interpolate Green',
            description: `Average of ${neighborData.gNeighbors.length} green neighbor${neighborData.gNeighbors.length > 1 ? 's' : ''}`,
            formula: `Ĝ = (${gVals.join(' + ')}) / ${neighborData.gNeighbors.length}`,
            result: `${gVals.map(v => `(0, ${v}, 0)`).join(', ')} → ${to255(g)}`
          });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          const bVals = neighborData.bNeighbors.map(n => to255(n.val));
        breakdown.push({
            step: '3. Interpolate Blue',
            description: `Average of ${neighborData.bNeighbors.length} blue neighbor${neighborData.bNeighbors.length > 1 ? 's' : ''}`,
            formula: `B̂ = (${bVals.join(' + ')}) / ${neighborData.bNeighbors.length}`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
          });
        }
      } else { // Blue
        if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0) {
          const gVals = neighborData.gNeighbors.map(n => to255(n.val));
        breakdown.push({
            step: '2. Interpolate Green',
            description: `Average of ${neighborData.gNeighbors.length} green neighbor${neighborData.gNeighbors.length > 1 ? 's' : ''}`,
            formula: `Ĝ = (${gVals.join(' + ')}) / ${neighborData.gNeighbors.length}`,
            result: `${gVals.map(v => `(0, ${v}, 0)`).join(', ')} → ${to255(g)}`
          });
        }
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          const rVals = neighborData.rNeighbors.map(n => to255(n.val));
        breakdown.push({
            step: '3. Interpolate Red',
            description: `Average of ${neighborData.rNeighbors.length} red neighbor${neighborData.rNeighbors.length > 1 ? 's' : ''}`,
            formula: `R̂ = (${rVals.join(' + ')}) / ${neighborData.rNeighbors.length}`,
            result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
          });
        }
      }
    } else if (algorithm === 'niu_edge_sensing') {
      const threshold = params?.niuLogisticThreshold ?? 0.1;
      const steepness = params?.niuLogisticSteepness;
      const vars = computeDirectionalVariations(input.cfaData, width, height, gx, gy, getChannel, getVal);
      const wH = logisticFunction(vars.horizontal, threshold, steepness);
      const wV = logisticFunction(vars.vertical, threshold, steepness);
      
      breakdown.push({
        step: '2. Compute Directional Variations',
        description: 'Measure edge strength in horizontal and vertical directions',
        formula: `Δ_H = |I(${gx+1}, ${gy}) - I(${gx-1}, ${gy})|, Δ_V = |I(${gx}, ${gy+1}) - I(${gx}, ${gy-1})|`,
        result: `Δ_H = ${vars.horizontal.toFixed(4)}, Δ_V = ${vars.vertical.toFixed(4)}`
      });
      
      breakdown.push({
        step: '3. Apply Logistic Function',
        description: `Compute edge weights using threshold θ = ${threshold}`,
        formula: `w = 1 / (1 + exp(-k(Δ - θ)))`,
        result: `w_H = ${wH.toFixed(4)}, w_V = ${wV.toFixed(4)}`
      });
      
      // Always show normalized edge weights after computing wH and wV
        const sumW = wH + wV;
        const nH = (sumW > 0) ? (1.0 - wH / sumW) : 0.5;
        const nV = (sumW > 0) ? (1.0 - wV / sumW) : 0.5;
      
      breakdown.push({
        step: '3a. Compute Normalized Edge Weights',
        description: 'Normalize weights to favor direction with lower edge strength',
        formula: `n_H = 1 - w_H/(w_H + w_V), n_V = 1 - w_V/(w_H + w_V)`,
        result: `n_H = ${nH.toFixed(4)}, n_V = ${nV.toFixed(4)}`
      });
      
      // Show green interpolation breakdown only when center is R or B (when we actually interpolate green)
      if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0 && (centerCh === 'r' || centerCh === 'b')) {
        // Separate horizontal and vertical green neighbors
        const greenH = neighborData.gNeighbors.filter(n => n.y === gy);
        const greenV = neighborData.gNeighbors.filter(n => n.x === gx);
        
        // Only show breakdown if we have both horizontal and vertical neighbors
        if (greenH.length > 0 || greenV.length > 0) {
          const gHVals = greenH.map(n => to255(n.val));
          const gVVals = greenV.map(n => to255(n.val));
          const gH = greenH.length > 0 ? greenH.reduce((sum, n) => sum + n.val, 0) / greenH.length : 0;
          const gV = greenV.length > 0 ? greenV.reduce((sum, n) => sum + n.val, 0) / greenV.length : 0;
          const gInterp = (gH * nH + gV * nV) / (nH + nV);
          
        breakdown.push({
          step: '4. Edge-Aware Green Interpolation',
            description: `Weighted average: ${greenH.length} horizontal green neighbors × n_H + ${greenV.length} vertical green neighbors × n_V`,
          formula: `Ĝ = (G_H × n_H + G_V × n_V) / (n_H + n_V)`,
            result: `G_H = avg(${gHVals.length > 0 ? gHVals.map(v => `(0, ${v}, 0)`).join(', ') : 'none'}) = (0, ${to255(gH)}, 0), G_V = avg(${gVVals.length > 0 ? gVVals.map(v => `(0, ${v}, 0)`).join(', ') : 'none'}) = (0, ${to255(gV)}, 0) → Ĝ = (0, ${to255(gInterp)}, 0)`
          });
        }
      }
      if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0 && centerCh === 'r') {
        const bVals = neighborData.bNeighbors.map(n => to255(n.val));
        const neighborType = cfaPattern === 'xtrans' ? 'blue neighbor' : 'blue corner neighbor';
        breakdown.push({
          step: '5. Interpolate Blue',
          description: cfaPattern === 'xtrans' 
            ? `Average of ${neighborData.bNeighbors.length} blue neighbors in 5×5 neighborhood`
            : `Using ${neighborData.bNeighbors.length} blue corner neighbor${neighborData.bNeighbors.length > 1 ? 's' : ''} via color difference`,
          formula: cfaPattern === 'xtrans' 
            ? `B̂ = average(B_neighbors)`
            : `B̂ = Ĝ + average(B - Ĝ)`,
          result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
        });
      }
      if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0 && centerCh === 'b') {
        const rVals = neighborData.rNeighbors.map(n => to255(n.val));
        breakdown.push({
          step: '5. Interpolate Red',
          description: cfaPattern === 'xtrans'
            ? `Average of ${neighborData.rNeighbors.length} red neighbors in 5×5 neighborhood`
            : `Using ${neighborData.rNeighbors.length} red corner neighbor${neighborData.rNeighbors.length > 1 ? 's' : ''} via color difference`,
          formula: cfaPattern === 'xtrans'
            ? `R̂ = average(R_neighbors)`
            : `R̂ = Ĝ + average(R - Ĝ)`,
          result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
        });
      }
      if ((neighborData.rNeighbors || neighborData.bNeighbors) && centerCh === 'g') {
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          const rVals = neighborData.rNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '2. Interpolate Red',
            description: `Average of ${neighborData.rNeighbors.length} red neighbor${neighborData.rNeighbors.length > 1 ? 's' : ''}`,
            formula: `R̂ = average(R_neighbors)`,
            result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
          });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          const bVals = neighborData.bNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '3. Interpolate Blue',
            description: `Average of ${neighborData.bNeighbors.length} blue neighbor${neighborData.bNeighbors.length > 1 ? 's' : ''}`,
            formula: `B̂ = average(B_neighbors)`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
          });
        }
      }
    } else if (algorithm === 'wu_polynomial') {
      const degree = params?.wuPolynomialDegree ?? 2;
      if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0 && centerCh !== 'g') {
        const gVals = neighborData.gNeighbors.map(n => to255(n.val));
      breakdown.push({
          step: '2. Polynomial Green Interpolation',
          description: `Degree-${degree} polynomial interpolation using ${neighborData.gNeighbors.length} green neighbors`,
          formula: `Ĝ = Σ(w_i × G_i) / Σ(w_i), where w_i = 1 / (1 + d_i^${degree})`,
          result: `${gVals.map(v => `(0, ${v}, 0)`).join(', ')} → ${to255(g)}`
        });
      }
      if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0 && centerCh === 'r') {
        const bVals = neighborData.bNeighbors.map(n => to255(n.val));
        breakdown.push({
          step: '3. Polynomial Blue Interpolation',
          description: `Degree-${degree} polynomial interpolation using ${neighborData.bNeighbors.length} blue neighbors`,
          formula: `B̂ = Σ(w_i × B_i) / Σ(w_i), where w_i = 1 / (1 + d_i^${degree})`,
          result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
        });
      }
      if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0 && centerCh === 'b') {
        const rVals = neighborData.rNeighbors.map(n => to255(n.val));
        breakdown.push({
          step: '3. Polynomial Red Interpolation',
          description: `Degree-${degree} polynomial interpolation using ${neighborData.rNeighbors.length} red neighbors`,
          formula: `R̂ = Σ(w_i × R_i) / Σ(w_i), where w_i = 1 / (1 + d_i^${degree})`,
          result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
        });
      }
      if ((neighborData.rNeighbors || neighborData.bNeighbors) && centerCh === 'g') {
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          const rVals = neighborData.rNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '2. Polynomial Red Interpolation',
            description: `Degree-${degree} polynomial interpolation using ${neighborData.rNeighbors.length} red neighbors`,
            formula: `R̂ = Σ(w_i × R_i) / Σ(w_i), where w_i = 1 / (1 + d_i^${degree})`,
            result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → ${to255(r)}`
          });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          const bVals = neighborData.bNeighbors.map(n => to255(n.val));
          breakdown.push({
            step: '3. Polynomial Blue Interpolation',
            description: `Degree-${degree} polynomial interpolation using ${neighborData.bNeighbors.length} blue neighbors`,
            formula: `B̂ = Σ(w_i × B_i) / Σ(w_i), where w_i = 1 / (1 + d_i^${degree})`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → ${to255(b)}`
          });
        }
      }
    } else if (algorithm === 'lien_edge_based') {
      // Compute edge direction for all cases
      const diffH = Math.abs(getVal(gx - 1, gy) - getVal(gx + 1, gy));
      const diffV = Math.abs(getVal(gx, gy - 1) - getVal(gx, gy + 1));
      // If diffH < diffV, horizontal variation is less (horizontal is smoother), so edge is horizontal
      // If diffV < diffH, vertical variation is less (vertical is smoother), so edge is vertical
      const edgeDirection = diffH < diffV ? 'horizontal' : 'vertical';
      // We interpolate along the smoother direction (same as edge direction in this interpretation)
      const interpolateDir = diffH < diffV ? 'horizontal' : 'vertical';
      
      breakdown.push({
        step: '2. Detect Edge Direction',
        description: 'Compare horizontal and vertical intensity differences to determine edge orientation',
        formula: `Δ_H = |I(${gx-1}, ${gy}) - I(${gx+1}, ${gy})|, Δ_V = |I(${gx}, ${gy-1}) - I(${gx}, ${gy+1})|`,
        result: `Δ_H = ${diffH.toFixed(4)}, Δ_V = ${diffV.toFixed(4)} → **Edge is ${edgeDirection}, interpolate along ${interpolateDir} direction** (lower variation = smoother)`
      });
      
      if (neighborData.gNeighbors && neighborData.gNeighbors.length > 0 && centerCh !== 'g') {
        // Separate neighbors by direction
        const gNeighborsH = neighborData.gNeighbors.filter(n => n.y === gy);
        const gNeighborsV = neighborData.gNeighbors.filter(n => n.x === gx);
        // If diffH < diffV, horizontal is smoother, so interpolate along horizontal direction (use horizontal neighbors)
        // If diffV < diffH, vertical is smoother, so interpolate along vertical direction (use vertical neighbors)
        // We interpolate along the smoother direction (perpendicular to the edge)
        const usedNeighbors = diffH < diffV ? gNeighborsH : gNeighborsV;
        const usedDir = diffH < diffV ? 'horizontal' : 'vertical';
        const gVals = usedNeighbors.map(n => to255(n.val));
        const avgG = usedNeighbors.length > 0 ? usedNeighbors.reduce((sum, n) => sum + n.val, 0) / usedNeighbors.length : 0;
        
        breakdown.push({
          step: '3. Edge-Aware Green Interpolation',
          description: `**Using ${usedNeighbors.length} green neighbor${usedNeighbors.length > 1 ? 's' : ''} in ${usedDir} direction** (${diffH < diffV ? 'horizontal' : 'vertical'} is smoother, edge is ${edgeDirection})`,
          formula: `Ĝ = average(G_neighbors in ${usedDir} direction)`,
          result: `${gVals.map(v => `(0, ${v}, 0)`).join(', ')} → (0, ${to255(avgG)}, 0)`
        });
      }
      if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0 && centerCh === 'r') {
        const bVals = neighborData.bNeighbors.map(n => to255(n.val));
        // For color difference, we need the average G from all 4 neighbors
        const allGNeighbors = [
          {x: gx-1, y: gy, val: getVal(gx-1, gy)},
          {x: gx+1, y: gy, val: getVal(gx+1, gy)},
          {x: gx, y: gy-1, val: getVal(gx, gy-1)},
          {x: gx, y: gy+1, val: getVal(gx, gy+1)}
        ].filter(n => getChannel(n.x, n.y) === 'g');
        const avgGAll = allGNeighbors.length > 0 ? allGNeighbors.reduce((sum, n) => sum + n.val, 0) / allGNeighbors.length : g;
        const bMinusG = neighborData.bNeighbors.map(n => n.val - avgGAll);
        const avgBMinusG = bMinusG.reduce((a, b) => a + b, 0) / bMinusG.length;
        const finalB = g + avgBMinusG;
        
        breakdown.push({
          step: '4. Interpolate Blue via Color Difference',
          description: `Using ${neighborData.bNeighbors.length} blue corner neighbor${neighborData.bNeighbors.length > 1 ? 's' : ''} and color difference assumption`,
          formula: `B̂ = Ĝ + average(B_i - G_avg)`,
          result: `B_i: ${bVals.map(v => `(0, 0, ${v})`).join(', ')}, G_avg = ${to255(avgGAll)}, B-G: ${bMinusG.map(v => to255(v)).join(', ')} → B̂ = ${to255(finalB)}`
        });
      }
      if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0 && centerCh === 'b') {
        const rVals = neighborData.rNeighbors.map(n => to255(n.val));
        // For color difference, we need the average G from all 4 neighbors
        const allGNeighbors = [
          {x: gx-1, y: gy, val: getVal(gx-1, gy)},
          {x: gx+1, y: gy, val: getVal(gx+1, gy)},
          {x: gx, y: gy-1, val: getVal(gx, gy-1)},
          {x: gx, y: gy+1, val: getVal(gx, gy+1)}
        ].filter(n => getChannel(n.x, n.y) === 'g');
        const avgGAll = allGNeighbors.length > 0 ? allGNeighbors.reduce((sum, n) => sum + n.val, 0) / allGNeighbors.length : g;
        const rMinusG = neighborData.rNeighbors.map(n => n.val - avgGAll);
        const avgRMinusG = rMinusG.reduce((a, b) => a + b, 0) / rMinusG.length;
        const finalR = g + avgRMinusG;
        
        breakdown.push({
          step: '4. Interpolate Red via Color Difference',
          description: `Using ${neighborData.rNeighbors.length} red corner neighbor${neighborData.rNeighbors.length > 1 ? 's' : ''} and color difference assumption`,
          formula: `R̂ = Ĝ + average(R_i - G_avg)`,
          result: `R_i: ${rVals.map(v => `(${v}, 0, 0)`).join(', ')}, G_avg = ${to255(avgGAll)}, R-G: ${rMinusG.map(v => to255(v)).join(', ')} → R̂ = ${to255(finalR)}`
        });
      }
      if ((neighborData.rNeighbors || neighborData.bNeighbors) && centerCh === 'g') {
        const leftCh = getChannel(gx - 1, gy);
        const isRedRow = (leftCh === 'r' || getChannel(gx + 1, gy) === 'r');
        
        if (neighborData.rNeighbors && neighborData.rNeighbors.length > 0) {
          // Separate R neighbors by direction based on edge and row type
          const rNeighborsH = neighborData.rNeighbors.filter(n => n.y === gy);
          const rNeighborsV = neighborData.rNeighbors.filter(n => n.x === gx);
          let usedRNeighbors: Array<{x: number, y: number, val: number}> = [];
          if (isRedRow) {
            usedRNeighbors = diffH < diffV ? rNeighborsH : rNeighborsV;
          } else {
            usedRNeighbors = diffH < diffV ? rNeighborsV : rNeighborsH;
          }
          const rVals = usedRNeighbors.map(n => to255(n.val));
          const avgR = usedRNeighbors.length > 0 ? usedRNeighbors.reduce((sum, n) => sum + n.val, 0) / usedRNeighbors.length : 0;
          
        const rDir = isRedRow ? (diffH < diffV ? 'horizontal' : 'vertical') : (diffH < diffV ? 'vertical' : 'horizontal');
        breakdown.push({
          step: '3. Edge-Aware Red Interpolation',
          description: `**Using ${usedRNeighbors.length} red neighbor${usedRNeighbors.length > 1 ? 's' : ''} in ${rDir} direction** (${isRedRow ? 'red' : 'blue'} row, ${edgeDirection} edge)`,
          formula: `R̂ = average(R_neighbors in ${rDir} direction)`,
          result: `${rVals.map(v => `(${v}, 0, 0)`).join(', ')} → (${to255(avgR)}, 0, 0)`
        });
        }
        if (neighborData.bNeighbors && neighborData.bNeighbors.length > 0) {
          // Separate B neighbors by direction based on edge and row type
          const bNeighborsH = neighborData.bNeighbors.filter(n => n.y === gy);
          const bNeighborsV = neighborData.bNeighbors.filter(n => n.x === gx);
          let usedBNeighbors: Array<{x: number, y: number, val: number}> = [];
          if (isRedRow) {
            usedBNeighbors = diffH < diffV ? bNeighborsV : bNeighborsH;
          } else {
            usedBNeighbors = diffH < diffV ? bNeighborsH : bNeighborsV;
          }
          const bVals = usedBNeighbors.map(n => to255(n.val));
          const avgB = usedBNeighbors.length > 0 ? usedBNeighbors.reduce((sum, n) => sum + n.val, 0) / usedBNeighbors.length : 0;
          
          const bDir = isRedRow ? (diffH < diffV ? 'vertical' : 'horizontal') : (diffH < diffV ? 'horizontal' : 'vertical');
          breakdown.push({
            step: '4. Edge-Aware Blue Interpolation',
            description: `**Using ${usedBNeighbors.length} blue neighbor${usedBNeighbors.length > 1 ? 's' : ''} in ${bDir} direction** (${isRedRow ? 'red' : 'blue'} row, ${edgeDirection} edge)`,
            formula: `B̂ = average(B_neighbors in ${bDir} direction)`,
            result: `${bVals.map(v => `(0, 0, ${v})`).join(', ')} → (0, 0, ${to255(avgB)})`
          });
        }
      }
    } else if (algorithm === 'kiku_residual' && kikuResidualDetail) {
      const {
        initialR,
        initialG,
        initialB,
        residualSamplesR,
        residualSamplesG,
        residualSamplesB,
        residualApplied,
        iterations,
      } = kikuResidualDetail;

      let stepIndex = 2;
      const formatResidual = (val: number) => {
        const scaled = val * 255;
        const sign = scaled >= 0 ? "+" : "−";
        const mag = Math.abs(scaled).toFixed(2);
        return `${sign}${mag}`;
      };

      breakdown.push({
        step: `${stepIndex++}. Bilinear Baseline`,
        description: 'Initial estimate using simple bilinear interpolation for all channels.',
        formula: '\\hat{I}_0 = \\text{Bilinear}(\\text{CFA})',
        result: `R₀=${to255(initialR)}, G₀=${to255(initialG)}, B₀=${to255(initialB)}`
      });

      const describeResiduals = (
        label: 'R' | 'G' | 'B',
        samples: Array<{ residual: number, val: number }>,
        applied: number,
        initialVal: number
      ) => {
        const sampleText = samples.length > 0
          ? `${samples.length} residual sample${samples.length > 1 ? 's' : ''}`
          : 'no nearby residual samples (residual = 0)';
        const meanFromSamples = samples.length > 0
          ? samples.reduce((sum, s) => sum + s.residual, 0) / samples.length
          : 0;
        const applied255 = applied * 255;
        const meanSamples255 = meanFromSamples * 255;
        breakdown.push({
          step: `${stepIndex++}. ${label} Residual Field`,
          description: `${sampleText}; applied residual ${formatResidual(applied)} (mean from samples: ${meanSamples255 >= 0 ? '+' : '−'}${Math.abs(meanSamples255).toFixed(2)}) in 0–255 units`,
          formula: `\\hat{${label}} = ${label}_0 + \\operatorname{mean}\\left(R_{${label}}\\right)`,
          result: `${label}̂ = ${to255(initialVal)} ${formatResidual(applied)} → ${to255(initialVal + applied)}`
        });
      };

      describeResiduals('R', residualSamplesR, residualApplied.r, initialR);
      describeResiduals('G', residualSamplesG, residualApplied.g, initialG);
      describeResiduals('B', residualSamplesB, residualApplied.b, initialB);

      if (iterations > 1) {
        breakdown.push({
          step: `${stepIndex++}. Extra Iterations`,
          description: `Residuals can be re-interpolated ${iterations - 1} more time(s) for refinement.`,
          formula: 'Repeat: R = I_{\\text{obs}} - \\hat{I},\\; \\hat{I} \\leftarrow \\hat{I} + \\text{Interp}(R)',
          result: `Î after ${iterations} iteration(s)`
        });
      }
    }
    
    breakdown.push({
      step: 'Final RGB',
      description: 'Final reconstructed color values',
      formula: `RGB(${gx}, ${gy})`,
      result: `(${to255(r)}, ${to255(g)}, ${to255(b)})`
    });
    
    return breakdown;
  }, [globalCursorX, globalCursorY, algorithm, params, input]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
         {/* Input Canvas */}
         <div className="space-y-1">
            <div className="text-xs font-semibold text-muted-foreground">CFA Input</div>
            <div className="relative aspect-square border rounded overflow-hidden cursor-crosshair"
                 onMouseDown={handleMouseDown}
                 onMouseMove={handleMouseMove}
                 onMouseUp={() => setIsDragging(false)}
                 onMouseLeave={() => setIsDragging(false)}>
                <canvas ref={inputCanvasRef} className="w-full h-full" style={{ imageRendering: 'pixelated' }} />
                <canvas ref={inputOverlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            </div>
         </div>
         
         {/* Output Canvas */}
         <div className="space-y-1">
            <div className="text-xs font-semibold text-muted-foreground">Demosaiced Output</div>
            <div className="relative aspect-square border rounded overflow-hidden">
                <canvas ref={outputCanvasRef} className="w-full h-full" style={{ imageRendering: 'pixelated' }} />
                <canvas ref={outputOverlayRef} className="absolute inset-0 w-full h-full pointer-events-none" />
            </div>
         </div>
      </div>
      
      {/* Hyperparameters Section */}
      {hyperparameters.length > 0 && (
        <div className="bg-muted/30 p-3 rounded-md border border-border/50">
          <div className="text-xs font-semibold text-primary mb-2">Algorithm Hyperparameters</div>
          <div className="space-y-1">
            {hyperparameters.map((param, idx) => (
              <div key={idx} className="flex justify-between items-center text-xs">
                <span className="text-muted-foreground">{param.label}:</span>
                <span className="font-mono font-medium">{param.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Calculation Breakdown Section */}
      <div className="bg-muted/30 p-3 rounded-md border border-border/50">
        <div className="text-xs font-semibold text-primary mb-2">Full Calculation for Pixel ({globalCursorX}, {globalCursorY})</div>
        <div className="space-y-3">
          {calculationBreakdown.map((step, idx) => (
            <div key={idx} className="space-y-1">
              <div className="text-xs font-medium text-foreground">{step.step}</div>
              <div className="text-[11px] text-muted-foreground">
                {step.description.split(/(\*\*.*?\*\*)/).map((part, i) => 
                  part.startsWith('**') && part.endsWith('**') ? (
                    <span key={i} className="font-bold text-foreground">{part.slice(2, -2)}</span>
                  ) : (
                    <span key={i}>{part}</span>
                  )
                )}
              </div>
              {step.formula && (
                <div className="text-[10px] font-mono bg-background/50 px-2 py-1 rounded border border-border/30">
                  <InlineMath math={step.formula} />
                </div>
              )}
              <div className="text-[11px] font-mono text-primary font-medium">
                = {step.result}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
