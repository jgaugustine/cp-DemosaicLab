import { DemosaicInput, DemosaicParams, ErrorStats, DemosaicAlgorithm } from '@/types/demosaic';
import { getBayerKernel, getXTransKernel } from './cfa';

const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));

// Helper to get channel function based on CFA pattern
const getChannelFunction = (input: DemosaicInput): (x: number, y: number) => 'r' | 'g' | 'b' => {
  if (input.cfaPattern === 'bayer') {
    return getBayerKernel(input.cfaPatternMeta.layout);
  } else if (input.cfaPattern === 'xtrans') {
    return getXTransKernel();
  }
  // Fallback (shouldn't happen for foveon in current implementation)
  return getBayerKernel('RGGB');
};

// Helper to safely get CFA value with mirror padding
const getCfaVal = (cfaData: Float32Array | Uint16Array, width: number, height: number, cx: number, cy: number) => {
  // Mirror padding for better edge handling
  let sx = cx; 
  if (sx < 0) sx = -sx; 
  else if (sx >= width) sx = 2*width - 2 - sx;
  
  let sy = cy; 
  if (sy < 0) sy = -sy; 
  else if (sy >= height) sy = 2*height - 2 - sy;
  
  // Clamp to bounds just in case mirror logic goes out for very small images
  sx = Math.max(0, Math.min(width - 1, sx));
  sy = Math.max(0, Math.min(height - 1, sy));
  
  return cfaData[sy * width + sx];
};

// Helper to collect neighbors of a specific color with expanding search radius
// Works for any CFA pattern by collecting all neighbors within maxRadius
const collectNeighbors = (
  cfaData: Float32Array | Uint16Array,
  width: number,
  height: number,
  x: number,
  y: number,
  targetColor: 'r' | 'g' | 'b',
  getChannel: (x: number, y: number) => 'r' | 'g' | 'b',
  maxRadius: number = 10
): { values: number[]; distances: number[] } => {
  const result: { values: number[]; distances: number[] } = { values: [], distances: [] };
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // Collect all neighbors within maxRadius (using L-infinity norm for efficiency)
  for (let dy = -maxRadius; dy <= maxRadius; dy++) {
    for (let dx = -maxRadius; dx <= maxRadius; dx++) {
      if (dx === 0 && dy === 0) continue;
      const ch = getChannel(x + dx, y + dy);
      if (ch === targetColor) {
        const dist = Math.sqrt(dx * dx + dy * dy);
        result.values.push(val(x + dx, y + dy));
        result.distances.push(dist);
      }
    }
  }
  
  return result;
};

// Helper functions for advanced algorithms
const logisticFunction = (x: number, threshold: number = 0.1, steepness?: number): number => {
  // Logistic function: 1 / (1 + exp(-k*(x - threshold)))
  // k controls steepness, threshold is the edge detection threshold
  // If steepness is provided, use it directly; otherwise, make k adaptive based on threshold
  const k = steepness !== undefined 
    ? steepness 
    : 20.0 / Math.max(0.01, threshold); // Higher k for lower thresholds = more sensitive
  return 1.0 / (1.0 + Math.exp(-k * (x - threshold)));
};

const computeDirectionalVariations = (
  cfaData: Float32Array | Uint16Array,
  width: number,
  height: number,
  x: number,
  y: number,
  getChannel: (x: number, y: number) => 'r' | 'g' | 'b'
): { horizontal: number; vertical: number; diagonal1: number; diagonal2: number } => {
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // Horizontal variation: |val(x+1,y) - val(x-1,y)|
  const hVar = Math.abs(val(x + 1, y) - val(x - 1, y));
  
  // Vertical variation: |val(x,y+1) - val(x,y-1)|
  const vVar = Math.abs(val(x, y + 1) - val(x, y - 1));
  
  // Diagonal variations
  const d1Var = Math.abs(val(x + 1, y + 1) - val(x - 1, y - 1));
  const d2Var = Math.abs(val(x + 1, y - 1) - val(x - 1, y + 1));
  
  return { horizontal: hVar, vertical: vVar, diagonal1: d1Var, diagonal2: d2Var };
};

const detectEdgeDirection = (
  cfaData: Float32Array | Uint16Array,
  width: number,
  height: number,
  x: number,
  y: number
): 'horizontal' | 'vertical' | 'diagonal1' | 'diagonal2' | 'none' => {
  const vars = computeDirectionalVariations(cfaData, width, height, x, y, () => 'g');
  
  const minVar = Math.min(vars.horizontal, vars.vertical, vars.diagonal1, vars.diagonal2);
  const threshold = 0.05;
  
  if (minVar > threshold) {
    // Strong edge detected, find direction with minimum variation (edge is along that direction)
    if (vars.horizontal === minVar) return 'horizontal';
    if (vars.vertical === minVar) return 'vertical';
    if (vars.diagonal1 === minVar) return 'diagonal1';
    if (vars.diagonal2 === minVar) return 'diagonal2';
  }
  
  return 'none';
};

const polynomialInterpolate = (
  values: number[],
  positions: number[],
  degree: number = 2
): number => {
  if (values.length === 0) return 0;
  if (values.length === 1) return values[0];
  
  // For degree 1 (linear), use simple average
  if (degree === 1) {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  
  // For degree 2+, use distance-weighted interpolation with polynomial weighting
  // Higher degree = more emphasis on closer neighbors
  const weights = positions.map((p, i) => {
    const dist = Math.max(0.1, p); // Avoid division by zero
    // Higher degree means steeper falloff with distance
    // For degree 2: 1/(1+d^2), for degree 3: 1/(1+d^3), etc.
    return 1.0 / (1.0 + Math.pow(dist, degree));
  });
  
  const sumW = weights.reduce((a, b) => a + b, 0);
  if (sumW === 0) {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  
  return values.reduce((sum, v, i) => sum + v * weights[i], 0) / sumW;
};

const computeResiduals = (
  observed: Float32Array,
  estimated: Float32Array,
  width: number,
  height: number
): Float32Array => {
  const residuals = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    residuals[i] = observed[i] - estimated[i];
  }
  return residuals;
};

// ... Bayer implementations (keep existing) ...
// Nearest Neighbor - works for any CFA pattern
export const demosaicNearest = (input: DemosaicInput): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getChannelFunction(input);
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerVal = cfaData[y * width + x];
      const centerCh = getChannel(x, y);
      const idx = (y * width + x) * 4;
      
      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'r') {
        r = centerVal;
        // Find nearest G and B with expanding search
        let foundG = false, foundB = false;
        for (let d = 1; d <= 10 && (!foundG || !foundB); d++) {
          for (let dy = -d; dy <= d && (!foundG || !foundB); dy++) {
            for (let dx = -d; dx <= d && (!foundG || !foundB); dx++) {
              if (dx === 0 && dy === 0) continue;
              const dist = Math.sqrt(dx * dx + dy * dy);
              // Only check pixels at current distance (not inner ones)
              if (dist > d - 1 && dist <= d) {
                const ch = getChannel(x + dx, y + dy);
                if (!foundG && ch === 'g') {
                  g = val(x + dx, y + dy);
                  foundG = true;
                }
                if (!foundB && ch === 'b') {
                  b = val(x + dx, y + dy);
                  foundB = true;
                }
              }
            }
          }
        }
      } else if (centerCh === 'b') {
        b = centerVal;
        // Find nearest G and R with expanding search
        let foundG = false, foundR = false;
        for (let d = 1; d <= 10 && (!foundG || !foundR); d++) {
          for (let dy = -d; dy <= d && (!foundG || !foundR); dy++) {
            for (let dx = -d; dx <= d && (!foundG || !foundR); dx++) {
              if (dx === 0 && dy === 0) continue;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist > d - 1 && dist <= d) {
                const ch = getChannel(x + dx, y + dy);
                if (!foundG && ch === 'g') {
                  g = val(x + dx, y + dy);
                  foundG = true;
                }
                if (!foundR && ch === 'r') {
                  r = val(x + dx, y + dy);
                  foundR = true;
                }
              }
            }
          }
        }
      } else {
        // Green pixel
        g = centerVal;
        // Find nearest R and B with expanding search
        let foundR = false, foundB = false;
        for (let d = 1; d <= 10 && (!foundR || !foundB); d++) {
          for (let dy = -d; dy <= d && (!foundR || !foundB); dy++) {
            for (let dx = -d; dx <= d && (!foundR || !foundB); dx++) {
              if (dx === 0 && dy === 0) continue;
              const dist = Math.sqrt(dx * dx + dy * dy);
              if (dist > d - 1 && dist <= d) {
                const ch = getChannel(x + dx, y + dy);
                if (!foundR && ch === 'r') {
                  r = val(x + dx, y + dy);
                  foundR = true;
                }
                if (!foundB && ch === 'b') {
                  b = val(x + dx, y + dy);
                  foundB = true;
                }
              }
            }
          }
        }
      }
      
      output.data[idx] = clamp(r);
      output.data[idx+1] = clamp(g);
      output.data[idx+2] = clamp(b);
      output.data[idx+3] = 255;
    }
  }
  return output;
};

// Bilinear Interpolation - works for any CFA pattern
export const demosaicBilinear = (input: DemosaicInput): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getChannelFunction(input);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerVal = cfaData[y * width + x];
      const centerCh = getChannel(x, y);
      const idx = (y * width + x) * 4;
      
      let r = 0, g = 0, b = 0;

      if (centerCh === 'g') {
        g = centerVal;
        
        // Collect neighbors with expanding search - USE RADIUS 1 FOR BILINEAR
        const rNeighbors = collectNeighbors(cfaData, width, height, x, y, 'r', getChannel, 1);
        const bNeighbors = collectNeighbors(cfaData, width, height, x, y, 'b', getChannel, 1);
        r = rNeighbors.values.length > 0 ? rNeighbors.values.reduce((a, b) => a + b, 0) / rNeighbors.values.length : 0;
        b = bNeighbors.values.length > 0 ? bNeighbors.values.reduce((a, b) => a + b, 0) / bNeighbors.values.length : 0;
      } else if (centerCh === 'r') {
        r = centerVal;
        
        // Collect neighbors with expanding search - USE RADIUS 1 FOR BILINEAR
        const gNeighbors = collectNeighbors(cfaData, width, height, x, y, 'g', getChannel, 1);
        const bNeighbors = collectNeighbors(cfaData, width, height, x, y, 'b', getChannel, 1);
        g = gNeighbors.values.length > 0 ? gNeighbors.values.reduce((a, b) => a + b, 0) / gNeighbors.values.length : 0;
        b = bNeighbors.values.length > 0 ? bNeighbors.values.reduce((a, b) => a + b, 0) / bNeighbors.values.length : 0;
      } else if (centerCh === 'b') {
        b = centerVal;
        
        // Collect neighbors with expanding search - USE RADIUS 1 FOR BILINEAR
        const gNeighbors = collectNeighbors(cfaData, width, height, x, y, 'g', getChannel, 1);
        const rNeighbors = collectNeighbors(cfaData, width, height, x, y, 'r', getChannel, 1);
        g = gNeighbors.values.length > 0 ? gNeighbors.values.reduce((a, b) => a + b, 0) / gNeighbors.values.length : 0;
        r = rNeighbors.values.length > 0 ? rNeighbors.values.reduce((a, b) => a + b, 0) / rNeighbors.values.length : 0;
      }
      
      output.data[idx] = clamp(r);
      output.data[idx+1] = clamp(g);
      output.data[idx+2] = clamp(b);
      output.data[idx+3] = 255;
    }
  }
  return output;
};

// Low-cost Edge Sensing - works for any CFA pattern
export const demosaicNiuEdgeSensing = (input: DemosaicInput, params?: DemosaicParams): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getChannelFunction(input);
  const threshold = params?.niuLogisticThreshold ?? 0.1;
  const steepness = params?.niuLogisticSteepness;
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // First pass: Interpolate green channel with edge awareness
  const greenInterp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      
      if (centerCh === 'g') {
        greenInterp[y * width + x] = centerVal;
      } else {
        // Use edge-aware interpolation for green at R/B pixels
        const vars = computeDirectionalVariations(cfaData, width, height, x, y, getChannel);
        const wH = logisticFunction(vars.horizontal, threshold, steepness);
        const wV = logisticFunction(vars.vertical, threshold, steepness);
        const sumW = wH + wV;
        const nH = (sumW > 0) ? (1.0 - wH / sumW) : 0.5;
        const nV = (sumW > 0) ? (1.0 - wV / sumW) : 0.5;
        
        // Get horizontal and vertical green neighbors
        const gH = (val(x - 1, y) + val(x + 1, y)) / 2;
        const gV = (val(x, y - 1) + val(x, y + 1)) / 2;
        
        greenInterp[y * width + x] = (gH * nH + gV * nV) / (nH + nV);
      }
    }
  }
  
  // Second pass: Interpolate R/B channels using color difference
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      const idx = (y * width + x) * 4;
      
      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'r') {
        r = centerVal;
        g = greenInterp[y * width + x];
        
        // Interpolate blue via color difference
        const bMinusG = [
          val(x - 1, y - 1) - greenInterp[(y - 1) * width + (x - 1)],
          val(x + 1, y - 1) - greenInterp[(y - 1) * width + (x + 1)],
          val(x - 1, y + 1) - greenInterp[(y + 1) * width + (x - 1)],
          val(x + 1, y + 1) - greenInterp[(y + 1) * width + (x + 1)]
        ];
        const avgBMinusG = bMinusG.reduce((a, b) => a + b, 0) / bMinusG.length;
        b = g + avgBMinusG;
      } else if (centerCh === 'b') {
        b = centerVal;
        g = greenInterp[y * width + x];
        
        // Interpolate red via color difference
        const rMinusG = [
          val(x - 1, y - 1) - greenInterp[(y - 1) * width + (x - 1)],
          val(x + 1, y - 1) - greenInterp[(y - 1) * width + (x + 1)],
          val(x - 1, y + 1) - greenInterp[(y + 1) * width + (x - 1)],
          val(x + 1, y + 1) - greenInterp[(y + 1) * width + (x + 1)]
        ];
        const avgRMinusG = rMinusG.reduce((a, b) => a + b, 0) / rMinusG.length;
        r = g + avgRMinusG;
      } else {
        // Green pixel
        g = centerVal;
        
        // Interpolate R/B based on row type
        const leftCh = getChannel(Math.max(0, x - 1), y);
        const isRedRow = (leftCh === 'r' || getChannel(Math.min(width - 1, x + 1), y) === 'r');
        
        if (isRedRow) {
          r = (val(x - 1, y) + val(x + 1, y)) / 2;
          b = (val(x, y - 1) + val(x, y + 1)) / 2;
        } else {
          r = (val(x, y - 1) + val(x, y + 1)) / 2;
          b = (val(x - 1, y) + val(x + 1, y)) / 2;
        }
      }
      
      output.data[idx] = clamp(r);
      output.data[idx + 1] = clamp(g);
      output.data[idx + 2] = clamp(b);
      output.data[idx + 3] = 255;
    }
  }
  
  return output;
};

// Hamilton-Adams - Efficient Edge-Based Technique - works for any CFA pattern
export const demosaicLienEdgeBased = (input: DemosaicInput): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getChannelFunction(input);
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // Helper to safely get pixel from array
  const getPixel = (arr: Float32Array | Uint16Array, x: number, y: number): number => {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    return arr[y * width + x];
  };
  
  // Allocate planes
  const G = new Float32Array(width * height);
  const RG = new Float32Array(width * height); // R - G
  const BG = new Float32Array(width * height); // B - G
  const RGComputed = new Uint8Array(width * height); // Flag: 1 if RG is known/computed
  const BGComputed = new Uint8Array(width * height); // Flag: 1 if BG is known/computed
  
  // Step 1: Initialize known samples from CFA
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ch = getChannel(x, y);
      const v = cfaData[y * width + x];
      if (ch === 'r') {
        G[y * width + x] = 0; // Will be interpolated
      } else if (ch === 'b') {
        G[y * width + x] = 0; // Will be interpolated
      } else {
        G[y * width + x] = v; // Green is known
      }
    }
  }
  
  // Step 2: Edge-oriented interpolation of G at R/B positions
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ch = getChannel(x, y);
      if (ch === 'r' || ch === 'b') {
        // Horizontal gradient using neighboring green samples
        // For Bayer: at R/B pixels, horizontal neighbors should be G
        const ghLeft = getChannel(x - 1, y) === 'g' ? getPixel(G, x - 1, y) : 0;
        const ghRight = getChannel(x + 1, y) === 'g' ? getPixel(G, x + 1, y) : 0;
        const gradH = Math.abs(ghLeft - ghRight);
        
        // Vertical gradient using neighboring green samples
        // For Bayer: at R/B pixels, vertical neighbors should be G
        const gvUp = getChannel(x, y - 1) === 'g' ? getPixel(G, x, y - 1) : 0;
        const gvDown = getChannel(x, y + 1) === 'g' ? getPixel(G, x, y + 1) : 0;
        const gradV = Math.abs(gvUp - gvDown);
        
        if (gradH < gradV && gvUp > 0 && gvDown > 0) {
          // Edge is stronger vertically, interpolate horizontally
          G[y * width + x] = (gvUp + gvDown) / 2;
        } else if (gradV < gradH && ghLeft > 0 && ghRight > 0) {
          // Edge stronger horizontally, interpolate vertically
          G[y * width + x] = (ghLeft + ghRight) / 2;
        } else {
          // No clear edge direction, isotropic - use available neighbors
          let sum = 0, count = 0;
          if (ghLeft > 0) { sum += ghLeft; count++; }
          if (ghRight > 0) { sum += ghRight; count++; }
          if (gvUp > 0) { sum += gvUp; count++; }
          if (gvDown > 0) { sum += gvDown; count++; }
          G[y * width + x] = count > 0 ? sum / count : 0;
        }
      }
    }
  }
  
  // Step 3: Build color difference planes R-G and B-G (known where R/B known)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ch = getChannel(x, y);
      const v = cfaData[y * width + x];
      const g = G[y * width + x];
      if (ch === 'r') {
        RG[y * width + x] = v - g;
        RGComputed[y * width + x] = 1;
      } else if (ch === 'b') {
        BG[y * width + x] = v - g;
        BGComputed[y * width + x] = 1;
      }
      // At green positions, RG/BG still 0 and not computed yet
    }
  }
  
  // Step 4: Edge-directed interpolation of RG (R-G) plane
  // Use multiple passes to handle cases where immediate neighbors aren't red
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const ch = getChannel(x, y);
        if (ch !== 'r' && RGComputed[y * width + x] === 0) {
          // Need to interpolate R-G at this pixel (green or blue position)
          const neighbors: {val: number, pos: string}[] = [];
          
          // Check immediate neighbors
          if (x > 0 && RGComputed[(y) * width + (x - 1)]) {
            neighbors.push({val: getPixel(RG, x - 1, y), pos: 'h'});
          }
          if (x < width - 1 && RGComputed[(y) * width + (x + 1)]) {
            neighbors.push({val: getPixel(RG, x + 1, y), pos: 'h'});
          }
          if (y > 0 && RGComputed[(y - 1) * width + (x)]) {
            neighbors.push({val: getPixel(RG, x, y - 1), pos: 'v'});
          }
          if (y < height - 1 && RGComputed[(y + 1) * width + (x)]) {
            neighbors.push({val: getPixel(RG, x, y + 1), pos: 'v'});
          }
          
          // In later passes, also try diagonals
          if (pass > 0) {
            if (x > 0 && y > 0 && RGComputed[(y - 1) * width + (x - 1)]) {
              neighbors.push({val: getPixel(RG, x - 1, y - 1), pos: 'd'});
            }
            if (x < width - 1 && y > 0 && RGComputed[(y - 1) * width + (x + 1)]) {
              neighbors.push({val: getPixel(RG, x + 1, y - 1), pos: 'd'});
            }
            if (x > 0 && y < height - 1 && RGComputed[(y + 1) * width + (x - 1)]) {
              neighbors.push({val: getPixel(RG, x - 1, y + 1), pos: 'd'});
            }
            if (x < width - 1 && y < height - 1 && RGComputed[(y + 1) * width + (x + 1)]) {
              neighbors.push({val: getPixel(RG, x + 1, y + 1), pos: 'd'});
            }
          }
          
          if (neighbors.length >= 2) {
            const hNeighbors = neighbors.filter(n => n.pos === 'h').map(n => n.val);
            const vNeighbors = neighbors.filter(n => n.pos === 'v').map(n => n.val);
            
            let gradH = 0, gradV = 0;
            if (hNeighbors.length >= 2) {
              gradH = Math.abs(hNeighbors[0] - hNeighbors[1]);
            }
            if (vNeighbors.length >= 2) {
              gradV = Math.abs(vNeighbors[0] - vNeighbors[1]);
            }
            
            if (gradH < gradV && vNeighbors.length >= 2) {
              RG[y * width + x] = vNeighbors.reduce((a, b) => a + b, 0) / vNeighbors.length;
              RGComputed[y * width + x] = 1;
            } else if (gradV < gradH && hNeighbors.length >= 2) {
              RG[y * width + x] = hNeighbors.reduce((a, b) => a + b, 0) / hNeighbors.length;
              RGComputed[y * width + x] = 1;
            } else if (neighbors.length > 0) {
              // Isotropic
              RG[y * width + x] = neighbors.reduce((a, b) => a + b.val, 0) / neighbors.length;
              RGComputed[y * width + x] = 1;
            }
          } else if (pass === 2 && neighbors.length === 1) {
            // Final pass fallback: use single neighbor if available
            RG[y * width + x] = neighbors[0].val;
            RGComputed[y * width + x] = 1;
          } else if (pass === 2 && neighbors.length === 0) {
            // Final pass fallback: use 0 if no neighbors found (shouldn't happen in normal cases)
            RG[y * width + x] = 0;
            RGComputed[y * width + x] = 1;
          }
        }
      }
    }
  }
  
  // Step 5: Edge-directed interpolation of BG (B-G) plane
  // Use multiple passes to handle cases where immediate neighbors aren't blue
  for (let pass = 0; pass < 3; pass++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const ch = getChannel(x, y);
        if (ch !== 'b' && BGComputed[y * width + x] === 0) {
          // Need to interpolate B-G at this pixel (green or red position)
          const neighbors: {val: number, pos: string}[] = [];
          
          // Check immediate neighbors
          if (x > 0 && BGComputed[(y) * width + (x - 1)]) {
            neighbors.push({val: getPixel(BG, x - 1, y), pos: 'h'});
          }
          if (x < width - 1 && BGComputed[(y) * width + (x + 1)]) {
            neighbors.push({val: getPixel(BG, x + 1, y), pos: 'h'});
          }
          if (y > 0 && BGComputed[(y - 1) * width + (x)]) {
            neighbors.push({val: getPixel(BG, x, y - 1), pos: 'v'});
          }
          if (y < height - 1 && BGComputed[(y + 1) * width + (x)]) {
            neighbors.push({val: getPixel(BG, x, y + 1), pos: 'v'});
          }
          
          // In later passes, also try diagonals
          if (pass > 0) {
            if (x > 0 && y > 0 && BGComputed[(y - 1) * width + (x - 1)]) {
              neighbors.push({val: getPixel(BG, x - 1, y - 1), pos: 'd'});
            }
            if (x < width - 1 && y > 0 && BGComputed[(y - 1) * width + (x + 1)]) {
              neighbors.push({val: getPixel(BG, x + 1, y - 1), pos: 'd'});
            }
            if (x > 0 && y < height - 1 && BGComputed[(y + 1) * width + (x - 1)]) {
              neighbors.push({val: getPixel(BG, x - 1, y + 1), pos: 'd'});
            }
            if (x < width - 1 && y < height - 1 && BGComputed[(y + 1) * width + (x + 1)]) {
              neighbors.push({val: getPixel(BG, x + 1, y + 1), pos: 'd'});
            }
          }
          
          if (neighbors.length >= 2) {
            const hNeighbors = neighbors.filter(n => n.pos === 'h').map(n => n.val);
            const vNeighbors = neighbors.filter(n => n.pos === 'v').map(n => n.val);
            
            let gradH = 0, gradV = 0;
            if (hNeighbors.length >= 2) {
              gradH = Math.abs(hNeighbors[0] - hNeighbors[1]);
            }
            if (vNeighbors.length >= 2) {
              gradV = Math.abs(vNeighbors[0] - vNeighbors[1]);
            }
            
            if (gradH < gradV && vNeighbors.length >= 2) {
              BG[y * width + x] = vNeighbors.reduce((a, b) => a + b, 0) / vNeighbors.length;
              BGComputed[y * width + x] = 1;
            } else if (gradV < gradH && hNeighbors.length >= 2) {
              BG[y * width + x] = hNeighbors.reduce((a, b) => a + b, 0) / hNeighbors.length;
              BGComputed[y * width + x] = 1;
            } else if (neighbors.length > 0) {
              // Isotropic
              BG[y * width + x] = neighbors.reduce((a, b) => a + b.val, 0) / neighbors.length;
              BGComputed[y * width + x] = 1;
            }
          } else if (pass === 2 && neighbors.length === 1) {
            // Final pass fallback: use single neighbor if available
            BG[y * width + x] = neighbors[0].val;
            BGComputed[y * width + x] = 1;
          } else if (pass === 2 && neighbors.length === 0) {
            // Final pass fallback: use 0 if no neighbors found (shouldn't happen in normal cases)
            BG[y * width + x] = 0;
            BGComputed[y * width + x] = 1;
          }
        }
      }
    }
  }
  
  // Step 6: Reconstruct full R and B from RG, BG and G
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const g = G[y * width + x];
      const r = RG[y * width + x] + g;
      const b = BG[y * width + x] + g;
      
      output.data[idx] = clamp(r);
      output.data[idx + 1] = clamp(g);
      output.data[idx + 2] = clamp(b);
      output.data[idx + 3] = 255;
    }
  }
  
  return output;
};

// Polynomial Interpolation - works for any CFA pattern
export const demosaicWuPolynomial = (input: DemosaicInput, params?: DemosaicParams): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getChannelFunction(input);
  const degree = params?.wuPolynomialDegree ?? 2;
  
  // For BAYER, use radius=2 to enable distance weighting benefits
  // Radius=1 makes all neighbors equidistant, eliminating distance weighting advantage
  // Radius=2 provides neighbors at multiple distances (d=1, √2, 2, √5) allowing
  // closer neighbors to be weighted more heavily, which helps with edge handling
  // The theoretical bound (≥28 dB) assumes radius=1 for smooth images, but
  // practical performance on edges benefits from radius=2's distance weighting
  // For X-Trans, use larger radius (5-6) since pattern is 6x6 aperiodic
  const maxRadius = input.cfaPattern === 'bayer' ? 2 : 6;
  
  // First pass: Interpolate green channel
  const greenInterp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      
      if (centerCh === 'g') {
        greenInterp[y * width + x] = centerVal;
      } else {
        // Use polynomial interpolation for green - optimized radius for pattern
        const gNeighbors = collectNeighbors(cfaData, width, height, x, y, 'g', getChannel, maxRadius);
        if (gNeighbors.values.length > 0) {
          greenInterp[y * width + x] = polynomialInterpolate(gNeighbors.values, gNeighbors.distances, degree);
        } else {
          greenInterp[y * width + x] = centerVal;
        }
      }
    }
  }
  
  // Second pass: Interpolate R/B using polynomial interpolation
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      const idx = (y * width + x) * 4;
      
      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'r') {
        r = centerVal;
        g = greenInterp[y * width + x];
        
        // Use polynomial interpolation on all blue neighbors
        const bNeighbors = collectNeighbors(cfaData, width, height, x, y, 'b', getChannel, maxRadius);
        if (bNeighbors.values.length > 0) {
          b = polynomialInterpolate(bNeighbors.values, bNeighbors.distances, degree);
        } else {
          b = g;
        }
      } else if (centerCh === 'b') {
        b = centerVal;
        g = greenInterp[y * width + x];
        
        // Use polynomial interpolation on all red neighbors
        const rNeighbors = collectNeighbors(cfaData, width, height, x, y, 'r', getChannel, maxRadius);
        if (rNeighbors.values.length > 0) {
          r = polynomialInterpolate(rNeighbors.values, rNeighbors.distances, degree);
        } else {
          r = g;
        }
      } else {
        // Green pixel
        g = centerVal;
        
        // Use polynomial interpolation on all neighbors
        const rNeighbors = collectNeighbors(cfaData, width, height, x, y, 'r', getChannel, maxRadius);
        const bNeighbors = collectNeighbors(cfaData, width, height, x, y, 'b', getChannel, maxRadius);
        if (rNeighbors.values.length > 0) {
          r = polynomialInterpolate(rNeighbors.values, rNeighbors.distances, degree);
        }
        if (bNeighbors.values.length > 0) {
          b = polynomialInterpolate(bNeighbors.values, bNeighbors.distances, degree);
        }
      }
      
      output.data[idx] = clamp(r);
      output.data[idx + 1] = clamp(g);
      output.data[idx + 2] = clamp(b);
      output.data[idx + 3] = 255;
    }
  }
  
  return output;
};

// Helper to collect residual neighbors with expanding search
const collectResidualNeighbors = (
  residualArray: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  targetColor: 'r' | 'g' | 'b',
  getChannel: (x: number, y: number) => 'r' | 'g' | 'b',
  maxRadius: number = 10
): number[] => {
  const values: number[] = [];
  
  // Collect all neighbors within maxRadius
  for (let dy = -maxRadius; dy <= maxRadius; dy++) {
    for (let dx = -maxRadius; dx <= maxRadius; dx++) {
      if (dx === 0 && dy === 0) continue;
      const idx2 = (y + dy) * width + (x + dx);
      if (idx2 >= 0 && idx2 < width * height && getChannel(x + dx, y + dy) === targetColor) {
        values.push(residualArray[idx2]);
      }
    }
  }
  
  return values;
};

// Residual Interpolation - works for any CFA pattern
export const demosaicKikuResidual = (input: DemosaicInput, params?: DemosaicParams): ImageData => {
  const { width, height, cfaData } = input;
  const iterations = params?.kikuResidualIterations ?? 1;
  const getChannel = getChannelFunction(input);
  
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // Initial estimation using bilinear interpolation - compute directly in 0-1 range
  // to avoid precision loss from clamp/rounding
  const initialR = new Float32Array(width * height);
  const initialG = new Float32Array(width * height);
  const initialB = new Float32Array(width * height);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      const idx = y * width + x;
      
      if (centerCh === 'g') {
        initialG[idx] = centerVal;
        const rNeighbors = collectNeighbors(cfaData, width, height, x, y, 'r', getChannel, 1);
        const bNeighbors = collectNeighbors(cfaData, width, height, x, y, 'b', getChannel, 1);
        initialR[idx] = rNeighbors.values.length > 0 ? rNeighbors.values.reduce((a, b) => a + b, 0) / rNeighbors.values.length : 0;
        initialB[idx] = bNeighbors.values.length > 0 ? bNeighbors.values.reduce((a, b) => a + b, 0) / bNeighbors.values.length : 0;
      } else if (centerCh === 'r') {
        initialR[idx] = centerVal;
        const gNeighbors = collectNeighbors(cfaData, width, height, x, y, 'g', getChannel, 1);
        const bNeighbors = collectNeighbors(cfaData, width, height, x, y, 'b', getChannel, 1);
        initialG[idx] = gNeighbors.values.length > 0 ? gNeighbors.values.reduce((a, b) => a + b, 0) / gNeighbors.values.length : 0;
        initialB[idx] = bNeighbors.values.length > 0 ? bNeighbors.values.reduce((a, b) => a + b, 0) / bNeighbors.values.length : 0;
      } else {
        initialB[idx] = centerVal;
        const gNeighbors = collectNeighbors(cfaData, width, height, x, y, 'g', getChannel, 1);
        const rNeighbors = collectNeighbors(cfaData, width, height, x, y, 'r', getChannel, 1);
        initialG[idx] = gNeighbors.values.length > 0 ? gNeighbors.values.reduce((a, b) => a + b, 0) / gNeighbors.values.length : 0;
        initialR[idx] = rNeighbors.values.length > 0 ? rNeighbors.values.reduce((a, b) => a + b, 0) / rNeighbors.values.length : 0;
      }
    }
  }
  
  // Refine estimates: initial + interpolated residuals (with iterations)
  // Initialize arrays for residual computation
  const residualR = new Float32Array(width * height);
  const residualG = new Float32Array(width * height);
  const residualB = new Float32Array(width * height);
  const interpolatedResidualR = new Float32Array(width * height);
  const interpolatedResidualG = new Float32Array(width * height);
  const interpolatedResidualB = new Float32Array(width * height);
  let currentR = new Float32Array(initialR);
  let currentG = new Float32Array(initialG);
  let currentB = new Float32Array(initialB);
  
  // Use the SAME interpolation method for residuals as used in initial bilinear estimate
  // This ensures residuals are interpolated consistently with the initial estimate
  // BAYER bilinear uses radius 1, X-Trans basic uses radius 2
  const residualSearchRadius = input.cfaPattern === 'bayer' ? 1 : 2;
  
  for (let iter = 0; iter < iterations; iter++) {
    // Reset residual arrays for this iteration
    residualR.fill(0);
    residualG.fill(0);
    residualB.fill(0);
    interpolatedResidualR.fill(0);
    interpolatedResidualG.fill(0);
    interpolatedResidualB.fill(0);
    
    // Compute residuals from current estimate
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerCh = getChannel(x, y);
        const centerVal = cfaData[y * width + x];
        const idx = y * width + x;
        
        if (centerCh === 'r') {
          residualR[idx] = centerVal - currentR[idx];
        } else if (centerCh === 'g') {
          residualG[idx] = centerVal - currentG[idx];
        } else {
          residualB[idx] = centerVal - currentB[idx];
        }
      }
    }
    
    // Interpolate residuals (reuse the same interpolation logic)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerCh = getChannel(x, y);
        const idx = y * width + x;
        
        if (centerCh === 'r') {
          interpolatedResidualR[idx] = residualR[idx];
          
          // Average all neighbors with optimized search radius
          const gVals = collectResidualNeighbors(residualG, width, height, x, y, 'g', getChannel, residualSearchRadius);
          const bVals = collectResidualNeighbors(residualB, width, height, x, y, 'b', getChannel, residualSearchRadius);
          interpolatedResidualG[idx] = gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : 0;
          interpolatedResidualB[idx] = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
        } else if (centerCh === 'g') {
          interpolatedResidualG[idx] = residualG[idx];
          
          // Average all neighbors with optimized search radius
          const rVals = collectResidualNeighbors(residualR, width, height, x, y, 'r', getChannel, residualSearchRadius);
          const bVals = collectResidualNeighbors(residualB, width, height, x, y, 'b', getChannel, residualSearchRadius);
          interpolatedResidualR[idx] = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
          interpolatedResidualB[idx] = bVals.length > 0 ? bVals.reduce((a, b) => a + b, 0) / bVals.length : 0;
        } else {
          interpolatedResidualB[idx] = residualB[idx];
          
          // Average all neighbors with optimized search radius
          const gVals = collectResidualNeighbors(residualG, width, height, x, y, 'g', getChannel, residualSearchRadius);
          const rVals = collectResidualNeighbors(residualR, width, height, x, y, 'r', getChannel, residualSearchRadius);
          interpolatedResidualG[idx] = gVals.length > 0 ? gVals.reduce((a, b) => a + b, 0) / gVals.length : 0;
          interpolatedResidualR[idx] = rVals.length > 0 ? rVals.reduce((a, b) => a + b, 0) / rVals.length : 0;
        }
      }
    }
    
    // Update current estimate
    for (let i = 0; i < width * height; i++) {
      currentR[i] += interpolatedResidualR[i];
      currentG[i] += interpolatedResidualG[i];
      currentB[i] += interpolatedResidualB[i];
    }
    
    // Enforce constraint: sampled pixels must equal CFA value
    // This ensures residuals at sampled pixels are always 0 and corrections only affect interpolated channels
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerCh = getChannel(x, y);
        const centerVal = cfaData[y * width + x];
        const idx = y * width + x;
        
        if (centerCh === 'r') {
          currentR[idx] = centerVal;
        } else if (centerCh === 'g') {
          currentG[idx] = centerVal;
        } else {
          currentB[idx] = centerVal;
        }
      }
    }
  }
  
  // Final output
  const output = new ImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const outIdx = (y * width + x) * 4;
      
      output.data[outIdx] = clamp(currentR[idx]);
      output.data[outIdx + 1] = clamp(currentG[idx]);
      output.data[outIdx + 2] = clamp(currentB[idx]);
      output.data[outIdx + 3] = 255;
    }
  }
  
  return output;
};

export const demosaicXTransBasic = (input: DemosaicInput): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getXTransKernel();
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];

      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'g') {
        g = centerVal;
        // Basic: average of available neighbors in 5x5
        let rSum = 0, rCnt = 0;
        let bSum = 0, bCnt = 0;
        // Scan 3x3 for fast approximation
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const ch = getChannel(x+dx, y+dy);
            const v = val(x+dx, y+dy);
            if (ch === 'r') { rSum += v; rCnt++; }
            if (ch === 'b') { bSum += v; bCnt++; }
          }
        }
        r = rCnt > 0 ? rSum / rCnt : 0;
        b = bCnt > 0 ? bSum / bCnt : 0;
      } else if (centerCh === 'r') {
        r = centerVal;
        let gSum = 0, gCnt = 0;
        // G is always adjacent in XTrans
        if (getChannel(x-1, y) === 'g') { gSum += val(x-1, y); gCnt++; }
        if (getChannel(x+1, y) === 'g') { gSum += val(x+1, y); gCnt++; }
        if (getChannel(x, y-1) === 'g') { gSum += val(x, y-1); gCnt++; }
        if (getChannel(x, y+1) === 'g') { gSum += val(x, y+1); gCnt++; }
        g = gCnt > 0 ? gSum / gCnt : 0;
        
        let bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
             if (getChannel(x+dx, y+dy) === 'b') { bSum += val(x+dx, y+dy); bCnt++; }
          }
        }
        b = bCnt > 0 ? bSum / bCnt : 0;
      } else { // Blue
        b = centerVal;
        let gSum = 0, gCnt = 0;
        if (getChannel(x-1, y) === 'g') { gSum += val(x-1, y); gCnt++; }
        if (getChannel(x+1, y) === 'g') { gSum += val(x+1, y); gCnt++; }
        if (getChannel(x, y-1) === 'g') { gSum += val(x, y-1); gCnt++; }
        if (getChannel(x, y+1) === 'g') { gSum += val(x, y+1); gCnt++; }
        g = gCnt > 0 ? gSum / gCnt : 0;
        
        let rSum = 0, rCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
             if (getChannel(x+dx, y+dy) === 'r') { rSum += val(x+dx, y+dy); rCnt++; }
          }
        }
        r = rCnt > 0 ? rSum / rCnt : 0;
      }
      
      output.data[idx] = clamp(r);
      output.data[idx+1] = clamp(g);
      output.data[idx+2] = clamp(b);
      output.data[idx+3] = 255;
    }
  }
  return output;
};

// X-Trans variants of the new algorithms
export const demosaicXTransNiuEdgeSensing = (input: DemosaicInput, params?: DemosaicParams): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getXTransKernel();
  const threshold = params?.niuLogisticThreshold ?? 0.1;
  const steepness = params?.niuLogisticSteepness;
  
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // First pass: Interpolate green channel
  const greenInterp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      
      if (centerCh === 'g') {
        greenInterp[y * width + x] = centerVal;
      } else {
        // Use edge-aware interpolation for green
        const vars = computeDirectionalVariations(cfaData, width, height, x, y, getChannel);
        const wH = logisticFunction(vars.horizontal, threshold, steepness);
        const wV = logisticFunction(vars.vertical, threshold, steepness);
        const sumW = wH + wV;
        const nH = (sumW > 0) ? (1.0 - wH / sumW) : 0.5;
        const nV = (sumW > 0) ? (1.0 - wV / sumW) : 0.5;
        
        // Collect green neighbors in horizontal and vertical directions separately
        let gHSum = 0, gHCnt = 0, gVSum = 0, gVCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            if (getChannel(x + dx, y + dy) === 'g') {
              if (dy === 0) { // Horizontal direction
                gHSum += val(x + dx, y + dy);
                gHCnt++;
              }
              if (dx === 0) { // Vertical direction
                gVSum += val(x + dx, y + dy);
                gVCnt++;
              }
            }
          }
        }
        const gH = gHCnt > 0 ? gHSum / gHCnt : centerVal;
        const gV = gVCnt > 0 ? gVSum / gVCnt : centerVal;
        greenInterp[y * width + x] = (gH * nH + gV * nV) / (nH + nV);
      }
    }
  }
  
  // Second pass: Interpolate R/B channels
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      const idx = (y * width + x) * 4;
      
      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'r') {
        r = centerVal;
        g = greenInterp[y * width + x];
        
        // Interpolate blue
        let bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'b') {
              bSum += val(x + dx, y + dy);
              bCnt++;
            }
          }
        }
        b = bCnt > 0 ? bSum / bCnt : g;
      } else if (centerCh === 'b') {
        b = centerVal;
        g = greenInterp[y * width + x];
        
        // Interpolate red
        let rSum = 0, rCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'r') {
              rSum += val(x + dx, y + dy);
              rCnt++;
            }
          }
        }
        r = rCnt > 0 ? rSum / rCnt : g;
      } else {
        g = centerVal;
        
        // Interpolate R/B
        let rSum = 0, rCnt = 0, bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const ch = getChannel(x + dx, y + dy);
            const v = val(x + dx, y + dy);
            if (ch === 'r') { rSum += v; rCnt++; }
            if (ch === 'b') { bSum += v; bCnt++; }
          }
        }
        r = rCnt > 0 ? rSum / rCnt : 0;
        b = bCnt > 0 ? bSum / bCnt : 0;
      }
      
      output.data[idx] = clamp(r);
      output.data[idx + 1] = clamp(g);
      output.data[idx + 2] = clamp(b);
      output.data[idx + 3] = 255;
    }
  }
  
  return output;
};

export const demosaicXTransLienEdgeBased = (input: DemosaicInput): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getXTransKernel();
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];

      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'g') {
        g = centerVal;
        // Detect edge direction
        const diffH = Math.abs(val(x - 1, y) - val(x + 1, y));
        const diffV = Math.abs(val(x, y - 1) - val(x, y + 1));
        
        // Interpolate R/B
        let rSum = 0, rCnt = 0, bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const ch = getChannel(x + dx, y + dy);
            const v = val(x + dx, y + dy);
            if (ch === 'r') { rSum += v; rCnt++; }
            if (ch === 'b') { bSum += v; bCnt++; }
          }
        }
        r = rCnt > 0 ? rSum / rCnt : 0;
        b = bCnt > 0 ? bSum / bCnt : 0;
      } else if (centerCh === 'r') {
        r = centerVal;
        // Edge-aware green interpolation: compare GREEN channel values only
        const gH1 = getChannel(x - 1, y) === 'g' ? val(x - 1, y) : 0;
        const gH2 = getChannel(x + 1, y) === 'g' ? val(x + 1, y) : 0;
        const gV1 = getChannel(x, y - 1) === 'g' ? val(x, y - 1) : 0;
        const gV2 = getChannel(x, y + 1) === 'g' ? val(x, y + 1) : 0;
        
        const diffH = Math.abs(gH1 - gH2);
        const diffV = Math.abs(gV1 - gV2);
        
        let gSum = 0, gCnt = 0;
        if (diffH < diffV) {
          // Edge is horizontal, use horizontal green neighbors
          if (gH1 > 0) { gSum += gH1; gCnt++; }
          if (gH2 > 0) { gSum += gH2; gCnt++; }
        } else {
          // Edge is vertical, use vertical green neighbors
          if (gV1 > 0) { gSum += gV1; gCnt++; }
          if (gV2 > 0) { gSum += gV2; gCnt++; }
        }
        g = gCnt > 0 ? gSum / gCnt : ((gH1 + gH2 + gV1 + gV2) / 4);
        
        // Interpolate blue
        let bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'b') {
              bSum += val(x + dx, y + dy);
              bCnt++;
            }
          }
        }
        b = bCnt > 0 ? bSum / bCnt : g;
      } else {
        b = centerVal;
        // Edge-aware green interpolation: compare GREEN channel values only
        const gH1 = getChannel(x - 1, y) === 'g' ? val(x - 1, y) : 0;
        const gH2 = getChannel(x + 1, y) === 'g' ? val(x + 1, y) : 0;
        const gV1 = getChannel(x, y - 1) === 'g' ? val(x, y - 1) : 0;
        const gV2 = getChannel(x, y + 1) === 'g' ? val(x, y + 1) : 0;
        
        const diffH = Math.abs(gH1 - gH2);
        const diffV = Math.abs(gV1 - gV2);
        
        let gSum = 0, gCnt = 0;
        if (diffH < diffV) {
          // Edge is horizontal, use horizontal green neighbors
          if (gH1 > 0) { gSum += gH1; gCnt++; }
          if (gH2 > 0) { gSum += gH2; gCnt++; }
        } else {
          // Edge is vertical, use vertical green neighbors
          if (gV1 > 0) { gSum += gV1; gCnt++; }
          if (gV2 > 0) { gSum += gV2; gCnt++; }
        }
        g = gCnt > 0 ? gSum / gCnt : ((gH1 + gH2 + gV1 + gV2) / 4);
        
        // Interpolate red
        let rSum = 0, rCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'r') {
              rSum += val(x + dx, y + dy);
              rCnt++;
            }
          }
        }
        r = rCnt > 0 ? rSum / rCnt : g;
      }
      
      output.data[idx] = clamp(r);
      output.data[idx + 1] = clamp(g);
      output.data[idx + 2] = clamp(b);
      output.data[idx + 3] = 255;
    }
  }
  return output;
};

export const demosaicXTransWuPolynomial = (input: DemosaicInput, params?: DemosaicParams): ImageData => {
  const { width, height, cfaData } = input;
  const output = new ImageData(width, height);
  const getChannel = getXTransKernel();
  const degree = params?.wuPolynomialDegree ?? 2;
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);

  // First pass: Interpolate green
  const greenInterp = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      
      if (centerCh === 'g') {
        greenInterp[y * width + x] = centerVal;
      } else {
        // Use polynomial interpolation for green
        const neighbors: { v: number; d: number }[] = [];
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            if (getChannel(x + dx, y + dy) === 'g') {
              const dist = Math.sqrt(dx * dx + dy * dy);
              neighbors.push({ v: val(x + dx, y + dy), d: dist });
            }
          }
        }
        if (neighbors.length > 0) {
          const values = neighbors.map(n => n.v);
          const positions = neighbors.map(n => n.d);
          greenInterp[y * width + x] = polynomialInterpolate(values, positions, degree);
        } else {
          greenInterp[y * width + x] = centerVal;
        }
      }
    }
  }
  
  // Second pass: Interpolate R/B with edge classification
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      const idx = (y * width + x) * 4;
      
      let r = 0, g = 0, b = 0;
      
      if (centerCh === 'r') {
        r = centerVal;
        g = greenInterp[y * width + x];
        
        // Classify edge
        const colorDiffH = Math.abs(val(x - 1, y) - val(x + 1, y));
        const colorDiffV = Math.abs(val(x, y - 1) - val(x, y + 1));
        
        // Interpolate blue using polynomial
        const bNeighbors: { v: number; d: number }[] = [];
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'b') {
              const dist = Math.sqrt(dx * dx + dy * dy);
              bNeighbors.push({ v: val(x + dx, y + dy), d: dist });
            }
          }
        }
        if (bNeighbors.length > 0) {
          const bValues = bNeighbors.map(n => n.v);
          const bPositions = bNeighbors.map(n => n.d);
          b = polynomialInterpolate(bValues, bPositions, degree);
        } else {
          b = g;
        }
      } else if (centerCh === 'b') {
        b = centerVal;
        g = greenInterp[y * width + x];
        
        // Interpolate red using polynomial
        const rNeighbors: { v: number; d: number }[] = [];
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'r') {
              const dist = Math.sqrt(dx * dx + dy * dy);
              rNeighbors.push({ v: val(x + dx, y + dy), d: dist });
            }
          }
        }
        if (rNeighbors.length > 0) {
          const rValues = rNeighbors.map(n => n.v);
          const rPositions = rNeighbors.map(n => n.d);
          r = polynomialInterpolate(rValues, rPositions, degree);
        } else {
          r = g;
        }
      } else {
        g = centerVal;
        
        // Interpolate R/B using polynomial
        const rNeighbors: { v: number; d: number }[] = [];
        const bNeighbors: { v: number; d: number }[] = [];
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const ch = getChannel(x + dx, y + dy);
            const v = val(x + dx, y + dy);
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (ch === 'r') { rNeighbors.push({ v, d: dist }); }
            if (ch === 'b') { bNeighbors.push({ v, d: dist }); }
          }
        }
        if (rNeighbors.length > 0) {
          const rValues = rNeighbors.map(n => n.v);
          const rPositions = rNeighbors.map(n => n.d);
          r = polynomialInterpolate(rValues, rPositions, degree);
        } else {
          r = 0;
        }
        if (bNeighbors.length > 0) {
          const bValues = bNeighbors.map(n => n.v);
          const bPositions = bNeighbors.map(n => n.d);
          b = polynomialInterpolate(bValues, bPositions, degree);
        } else {
          b = 0;
        }
      }
      
      output.data[idx] = clamp(r);
      output.data[idx + 1] = clamp(g);
      output.data[idx + 2] = clamp(b);
      output.data[idx + 3] = 255;
    }
  }
  return output;
};

export const demosaicXTransKikuResidual = (input: DemosaicInput, params?: DemosaicParams): ImageData => {
  const { width, height, cfaData } = input;
  const iterations = params?.kikuResidualIterations ?? 1;
  const getChannel = getXTransKernel();
  const val = (cx: number, cy: number) => getCfaVal(cfaData, width, height, cx, cy);
  
  // Initial estimation using basic X-Trans - compute directly in 0-1 range
  // to avoid precision loss from clamp/rounding
  const initialR = new Float32Array(width * height);
  const initialG = new Float32Array(width * height);
  const initialB = new Float32Array(width * height);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const centerCh = getChannel(x, y);
      const centerVal = cfaData[y * width + x];
      const idx = y * width + x;
      
      if (centerCh === 'g') {
        initialG[idx] = centerVal;
        let rSum = 0, rCnt = 0, bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const ch = getChannel(x + dx, y + dy);
            const v = val(x + dx, y + dy);
            if (ch === 'r') { rSum += v; rCnt++; }
            if (ch === 'b') { bSum += v; bCnt++; }
          }
        }
        initialR[idx] = rCnt > 0 ? rSum / rCnt : 0;
        initialB[idx] = bCnt > 0 ? bSum / bCnt : 0;
      } else if (centerCh === 'r') {
        initialR[idx] = centerVal;
        let gSum = 0, gCnt = 0;
        if (getChannel(x - 1, y) === 'g') { gSum += val(x - 1, y); gCnt++; }
        if (getChannel(x + 1, y) === 'g') { gSum += val(x + 1, y); gCnt++; }
        if (getChannel(x, y - 1) === 'g') { gSum += val(x, y - 1); gCnt++; }
        if (getChannel(x, y + 1) === 'g') { gSum += val(x, y + 1); gCnt++; }
        initialG[idx] = gCnt > 0 ? gSum / gCnt : 0;
        
        let bSum = 0, bCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'b') { bSum += val(x + dx, y + dy); bCnt++; }
          }
        }
        initialB[idx] = bCnt > 0 ? bSum / bCnt : 0;
      } else {
        initialB[idx] = centerVal;
        let gSum = 0, gCnt = 0;
        if (getChannel(x - 1, y) === 'g') { gSum += val(x - 1, y); gCnt++; }
        if (getChannel(x + 1, y) === 'g') { gSum += val(x + 1, y); gCnt++; }
        if (getChannel(x, y - 1) === 'g') { gSum += val(x, y - 1); gCnt++; }
        if (getChannel(x, y + 1) === 'g') { gSum += val(x, y + 1); gCnt++; }
        initialG[idx] = gCnt > 0 ? gSum / gCnt : 0;
        
        let rSum = 0, rCnt = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            if (getChannel(x + dx, y + dy) === 'r') { rSum += val(x + dx, y + dy); rCnt++; }
          }
        }
        initialR[idx] = rCnt > 0 ? rSum / rCnt : 0;
      }
    }
  }
  
  // Refine estimates with iterations
  let currentR = new Float32Array(initialR);
  let currentG = new Float32Array(initialG);
  let currentB = new Float32Array(initialB);
  
  const residualR = new Float32Array(width * height);
  const residualG = new Float32Array(width * height);
  const residualB = new Float32Array(width * height);
  const interpolatedResidualR = new Float32Array(width * height);
  const interpolatedResidualG = new Float32Array(width * height);
  const interpolatedResidualB = new Float32Array(width * height);
  
  for (let iter = 0; iter < iterations; iter++) {
    // Reset residual arrays for this iteration
    residualR.fill(0);
    residualG.fill(0);
    residualB.fill(0);
    interpolatedResidualR.fill(0);
    interpolatedResidualG.fill(0);
    interpolatedResidualB.fill(0);
    
    // Compute residuals from current estimate
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerCh = getChannel(x, y);
        const centerVal = cfaData[y * width + x];
        const idx = y * width + x;
        
        if (centerCh === 'r') {
          residualR[idx] = centerVal - currentR[idx];
        } else if (centerCh === 'g') {
          residualG[idx] = centerVal - currentG[idx];
        } else {
          residualB[idx] = centerVal - currentB[idx];
        }
      }
    }
    
    // Interpolate residuals
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerCh = getChannel(x, y);
        const idx = y * width + x;
        
        if (centerCh === 'r') {
          interpolatedResidualR[idx] = residualR[idx];
          // Interpolate G and B residuals
          let gSum = 0, gCnt = 0, bSum = 0, bCnt = 0;
          for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              if (dx === 0 && dy === 0) continue;
              const ch = getChannel(x + dx, y + dy);
              if (ch === 'g') { gSum += residualG[(y + dy) * width + (x + dx)]; gCnt++; }
              if (ch === 'b') { bSum += residualB[(y + dy) * width + (x + dx)]; bCnt++; }
            }
          }
          interpolatedResidualG[idx] = gCnt > 0 ? gSum / gCnt : 0;
          interpolatedResidualB[idx] = bCnt > 0 ? bSum / bCnt : 0;
        } else if (centerCh === 'g') {
          interpolatedResidualG[idx] = residualG[idx];
          // Interpolate R and B residuals
          let rSum = 0, rCnt = 0, bSum = 0, bCnt = 0;
          for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              if (dx === 0 && dy === 0) continue;
              const ch = getChannel(x + dx, y + dy);
              if (ch === 'r') { rSum += residualR[(y + dy) * width + (x + dx)]; rCnt++; }
              if (ch === 'b') { bSum += residualB[(y + dy) * width + (x + dx)]; bCnt++; }
            }
          }
          interpolatedResidualR[idx] = rCnt > 0 ? rSum / rCnt : 0;
          interpolatedResidualB[idx] = bCnt > 0 ? bSum / bCnt : 0;
        } else {
          interpolatedResidualB[idx] = residualB[idx];
          // Interpolate R and G residuals
          let rSum = 0, rCnt = 0, gSum = 0, gCnt = 0;
          for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              if (dx === 0 && dy === 0) continue;
              const ch = getChannel(x + dx, y + dy);
              if (ch === 'r') { rSum += residualR[(y + dy) * width + (x + dx)]; rCnt++; }
              if (ch === 'g') { gSum += residualG[(y + dy) * width + (x + dx)]; gCnt++; }
            }
          }
          interpolatedResidualR[idx] = rCnt > 0 ? rSum / rCnt : 0;
          interpolatedResidualG[idx] = gCnt > 0 ? gSum / gCnt : 0;
        }
      }
    }
    
    // Update current estimate
    for (let i = 0; i < width * height; i++) {
      currentR[i] += interpolatedResidualR[i];
      currentG[i] += interpolatedResidualG[i];
      currentB[i] += interpolatedResidualB[i];
    }
    
    // Enforce constraint: sampled pixels must equal CFA value
    // This ensures residuals at sampled pixels are always 0 and corrections only affect interpolated channels
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerCh = getChannel(x, y);
        const centerVal = cfaData[y * width + x];
        const idx = y * width + x;
        
        if (centerCh === 'r') {
          currentR[idx] = centerVal;
        } else if (centerCh === 'g') {
          currentG[idx] = centerVal;
        } else {
          currentB[idx] = centerVal;
        }
      }
    }
  }
  
  // Final output
  const output = new ImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const outIdx = (y * width + x) * 4;
      
      output.data[outIdx] = clamp(currentR[idx]);
      output.data[outIdx + 1] = clamp(currentG[idx]);
      output.data[outIdx + 2] = clamp(currentB[idx]);
      output.data[outIdx + 3] = 255;
    }
  }
  
  return output;
};

// ... computeErrorStats ...
export const computeErrorStats = (
  original: ImageData,
  demosaiced: ImageData
): ErrorStats => {
  let mseR = 0, mseG = 0, mseB = 0;
  let maeR = 0, maeG = 0, maeB = 0;
  const l2Map = new Float32Array(original.width * original.height);
  
  for (let i = 0; i < original.data.length; i += 4) {
    const dr = demosaiced.data[i] - original.data[i];
    const dg = demosaiced.data[i+1] - original.data[i+1];
    const db = demosaiced.data[i+2] - original.data[i+2];
    
    const errSq = dr*dr + dg*dg + db*db;
    l2Map[i/4] = Math.sqrt(errSq);
    
    mseR += dr*dr;
    mseG += dg*dg;
    mseB += db*db;

    maeR += Math.abs(dr);
    maeG += Math.abs(dg);
    maeB += Math.abs(db);
  }
  
  const px = original.width * original.height;
  mseR /= px;
  mseG /= px;
  mseB /= px;
  const mseTotal = (mseR + mseG + mseB) / 3;

  maeR /= px;
  maeG /= px;
  maeB /= px;
  const maeTotal = (maeR + maeG + maeB) / 3;
  
  const psnr = (mse: number) => mse === 0 ? 100 : 10 * Math.log10((255*255) / mse);

  // Simple luminance-based SSIM over the full image (single-window approximation)
  // This is not patch-based, but gives a scalar structural similarity indicator.
  let muX = 0, muY = 0;
  let sigmaX2 = 0, sigmaY2 = 0, sigmaXY = 0;
  const L = 255;
  const C1 = (0.01 * L) ** 2;
  const C2 = (0.03 * L) ** 2;

  for (let i = 0; i < original.data.length; i += 4) {
    const xr = original.data[i];
    const xg = original.data[i+1];
    const xb = original.data[i+2];
    const yr = demosaiced.data[i];
    const yg = demosaiced.data[i+1];
    const yb = demosaiced.data[i+2];

    const xL = 0.299 * xr + 0.587 * xg + 0.114 * xb;
    const yL = 0.299 * yr + 0.587 * yg + 0.114 * yb;

    muX += xL;
    muY += yL;
  }

  const n = original.data.length / 4;
  if (n > 0) {
    muX /= n;
    muY /= n;

    for (let i = 0; i < original.data.length; i += 4) {
      const xr = original.data[i];
      const xg = original.data[i+1];
      const xb = original.data[i+2];
      const yr = demosaiced.data[i];
      const yg = demosaiced.data[i+1];
      const yb = demosaiced.data[i+2];

      const xL = 0.299 * xr + 0.587 * xg + 0.114 * xb;
      const yL = 0.299 * yr + 0.587 * yg + 0.114 * yb;

      const dx = xL - muX;
      const dy = yL - muY;
      sigmaX2 += dx * dx;
      sigmaY2 += dy * dy;
      sigmaXY += dx * dy;
    }

    sigmaX2 /= n;
    sigmaY2 /= n;
    sigmaXY /= n;
  }

  const ssim =
    ((2 * muX * muY + C1) * (2 * sigmaXY + C2)) /
    ((muX * muX + muY * muY + C1) * (sigmaX2 + sigmaY2 + C2));
  
  return {
    mse: { r: mseR, g: mseG, b: mseB, total: mseTotal },
    psnr: { r: psnr(mseR), g: psnr(mseG), b: psnr(mseB), total: psnr(mseTotal) },
    mae: { r: maeR, g: maeG, b: maeB, total: maeTotal },
    ssim,
    l2Map
  };
};

