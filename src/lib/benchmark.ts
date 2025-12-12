import { 
  DemosaicInput, 
  DemosaicAlgorithm, 
  DemosaicParams, 
  CFAType, 
  ErrorStats 
} from '@/types/demosaic';
import { BenchmarkResult, BenchmarkConfig } from '@/types/benchmark';
import { simulateCFA } from './cfa';
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
  computeErrorStats
} from './demosaic';

// Helper to run a single demosaic algorithm
const runDemosaic = (
  input: DemosaicInput, 
  algorithm: DemosaicAlgorithm, 
  params?: DemosaicParams
): ImageData => {
  // Use X-Trans specific implementations when CFA pattern is X-Trans
  if (input.cfaPattern === 'xtrans') {
    switch (algorithm) {
      case 'nearest':
        return demosaicNearest(input);
      case 'bilinear':
        return demosaicBilinear(input);
      case 'niu_edge_sensing':
        return demosaicXTransNiuEdgeSensing(input, params);
      case 'lien_edge_based':
        return demosaicXTransLienEdgeBased(input);
      case 'wu_polynomial':
        return demosaicXTransWuPolynomial(input, params);
      case 'kiku_residual':
        return demosaicXTransKikuResidual(input, params);
      default:
        return new ImageData(input.width, input.height);
    }
  }
  
  // Use Bayer/generic implementations for Bayer pattern
  switch (algorithm) {
    case 'nearest':
      return demosaicNearest(input);
    case 'bilinear':
      return demosaicBilinear(input);
    case 'niu_edge_sensing':
      return demosaicNiuEdgeSensing(input, params);
    case 'lien_edge_based':
      return demosaicLienEdgeBased(input);
    case 'wu_polynomial':
      return demosaicWuPolynomial(input, params);
    case 'kiku_residual':
      return demosaicKikuResidual(input, params);
    default:
      return new ImageData(input.width, input.height);
  }
};

// Get memory usage if available (Chrome only)
const getMemoryUsage = (): number | undefined => {
  if ('memory' in performance && (performance as any).memory) {
    const memory = (performance as any).memory;
    return memory.usedJSHeapSize / (1024 * 1024); // Convert to MB
  }
  return undefined;
};

// Calculate t-value for 95% confidence interval
// Using t-distribution values for small samples (approximation)
const getTValue95 = (n: number): number => {
  // For n >= 30, use normal approximation (1.96)
  if (n >= 30) return 1.96;
  // For smaller samples, approximate t-distribution
  // t-values for 95% CI: n=2:12.71, n=3:4.30, n=4:3.18, n=5:2.78, n=10:2.26, n=20:2.09, n=30:2.04
  const tTable: Record<number, number> = {
    2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
    11: 2.23, 12: 2.20, 13: 2.18, 14: 2.16, 15: 2.14, 16: 2.13, 17: 2.12, 18: 2.11, 19: 2.10, 20: 2.09,
    21: 2.08, 22: 2.07, 23: 2.07, 24: 2.06, 25: 2.06, 26: 2.06, 27: 2.05, 28: 2.05, 29: 2.05, 30: 2.04
  };
  if (n <= 30 && tTable[n]) return tTable[n];
  // Linear interpolation for values between table entries
  const lower = Math.floor(n);
  const upper = Math.ceil(n);
  if (lower < 2) return 12.71;
  if (upper > 30) return 1.96;
  if (tTable[lower] && tTable[upper]) {
    return tTable[lower] + (tTable[upper] - tTable[lower]) * (n - lower);
  }
  return 1.96; // Fallback
};

// Calculate statistics from an array of numbers
const calculateStats = (values: number[]): {
  average: number;
  median: number;
  min: number;
  max: number;
  stdDev: number;
  ci95Lower: number;
  ci95Upper: number;
} => {
  if (values.length === 0) {
    return { average: 0, median: 0, min: 0, max: 0, stdDev: 0, ci95Lower: 0, ci95Upper: 0 };
  }
  
  const sorted = [...values].sort((a, b) => a - b);
  const sum = values.reduce((a, b) => a + b, 0);
  const average = sum / values.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  
  const variance = values.reduce((acc, val) => acc + Math.pow(val - average, 2), 0) / values.length;
  const stdDev = Math.sqrt(variance);
  
  // Calculate 95% confidence interval
  const n = values.length;
  const tValue = getTValue95(n);
  const marginOfError = (tValue * stdDev) / Math.sqrt(n);
  const ci95Lower = average - marginOfError;
  const ci95Upper = average + marginOfError;
  
  return { average, median, min, max, stdDev, ci95Lower, ci95Upper };
};

// Run a single benchmark test
export const runSingleBenchmark = async (
  input: DemosaicInput,
  algorithm: DemosaicAlgorithm,
  imageName: string,
  iterations: number,
  enableQualityMetrics: boolean,
  params?: DemosaicParams,
  shouldCancel?: () => boolean
): Promise<BenchmarkResult> => {
  const pixelCount = input.width * input.height;
  const executionTimes: number[] = [];
  let memoryBefore: number | undefined;
  let memoryAfter: number | undefined;
  
  // Check for cancellation before starting
  if (shouldCancel && shouldCancel()) {
    throw new Error('Benchmark cancelled');
  }
  
  // Warm-up run (not counted)
  if (iterations > 1) {
    if (shouldCancel && shouldCancel()) {
      throw new Error('Benchmark cancelled');
    }
    runDemosaic(input, algorithm, params);
  }
  
  // Measure memory before
  if (iterations > 0) {
    memoryBefore = getMemoryUsage();
  }
  
  // Run iterations
  for (let i = 0; i < iterations; i++) {
    // Check for cancellation before each iteration
    if (shouldCancel && shouldCancel()) {
      throw new Error('Benchmark cancelled');
    }
    
    // Force garbage collection hint (may not work in all browsers)
    if (i > 0 && i % 10 === 0 && 'gc' in (globalThis as any)) {
      try {
        (globalThis as any).gc();
      } catch (e) {
        // GC not available
      }
    }
    
    const start = performance.now();
    const result = runDemosaic(input, algorithm, params);
    const end = performance.now();
    
    executionTimes.push(end - start);
    
    // Yield to allow cancellation check
    if (i < iterations - 1) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    
    // Store the last result for quality metrics
    if (i === iterations - 1) {
      // Will compute quality metrics below if needed
    }
  }
  
  // Measure memory after
  if (iterations > 0) {
    memoryAfter = getMemoryUsage();
  }
  
  const stats = calculateStats(executionTimes);
  const throughputMPs = pixelCount / (stats.average / 1000) / 1_000_000; // Megapixels per second
  
  // Calculate throughput CI bounds
  const throughputLower = pixelCount / (stats.ci95Upper / 1000) / 1_000_000;
  const throughputUpper = pixelCount / (stats.ci95Lower / 1000) / 1_000_000;
  
  // Check for cancellation before quality metrics
  if (shouldCancel && shouldCancel()) {
    throw new Error('Benchmark cancelled');
  }
  
  // Compute quality metrics if enabled and ground truth is available
  let quality: BenchmarkResult['quality'] | undefined;
  if (enableQualityMetrics && input.groundTruthRGB) {
    const result = runDemosaic(input, algorithm, params);
    const errorStats = computeErrorStats(input.groundTruthRGB, result);
    quality = {
      mse: errorStats.mse,
      psnr: errorStats.psnr,
      mae: errorStats.mae,
      ssim: errorStats.ssim,
      l2Map: errorStats.l2Map, // For heatmap visualization
    };
  }
  
  const memoryUsage = memoryBefore !== undefined && memoryAfter !== undefined
    ? Math.max(0, memoryAfter - memoryBefore)
    : undefined;
  
  return {
    id: `${algorithm}_${imageName}_${input.cfaPattern}_${Date.now()}`,
    algorithm,
    imageName,
    cfaPattern: input.cfaPattern,
    width: input.width,
    height: input.height,
    pixelCount,
    performance: {
      executionTimeMs: executionTimes,
      averageTimeMs: stats.average,
      medianTimeMs: stats.median,
      minTimeMs: stats.min,
      maxTimeMs: stats.max,
      stdDevTimeMs: stats.stdDev,
      ci95LowerMs: stats.ci95Lower,
      ci95UpperMs: stats.ci95Upper,
      throughputMPs,
      memoryUsageMB: memoryUsage,
    },
    quality,
    timestamp: Date.now(),
    iterations,
  };
};

// Generate a unique ID for benchmark results
let benchmarkCounter = 0;
export const generateBenchmarkId = (): string => {
  return `benchmark_${Date.now()}_${++benchmarkCounter}`;
};

