import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import {
  Calculator,
  Settings,
  Info,
  Play,
  Zap,
  Cpu,
  Eye,
  Grid,
  Layers
} from 'lucide-react';

const VisionTransformerAnalysis = () => {
  const [selectedParam, setSelectedParam] = useState('embedding_dim');
  const [showCalculation] = useState(true);
  const [animateCalculation, setAnimateCalculation] = useState(false);

  const [paramRanges, setParamRanges] = useState({
    embedding_dim: { min: 256, max: 2048, step: 256 },
    sequence_length: { min: 50, max: 1600, step: 100 },
    num_heads: { min: 4, max: 24, step: 4 },
    image_size: { min: 112, max: 512, step: 56 }
  });

  const [opticalCore, setOpticalCore] = useState({
    wavelength_channels: 32,
    microrings_per_channel: 64,
    clock_cycles_per_op: 1,
    parallel_ops: 32,
    energy_per_access: 0.1,
    area_per_core: 1.0,
    throughput_gops: 100
  });

  const [baseParams, setBaseParams] = useState({
    embedding_dim: 768,
    num_heads: 12,
    image_size: 224,
    patch_size: 16,
    sequence_length: 197
  });

  const [calculationStep, setCalculationStep] = useState(0);

  useEffect(() => {
    const newSeqLength = Math.pow(baseParams.image_size / baseParams.patch_size, 2) + 1;
    setBaseParams(prev => ({
      ...prev,
      sequence_length: Math.floor(newSeqLength)
    }));
  }, [baseParams.image_size, baseParams.patch_size]);

  const detailedCalc = useMemo(() => calculateAccessesDetailed(baseParams), [baseParams]);
  const opticalMetrics = useMemo(
    () => calculateOpticalMetrics(detailedCalc.totalAccesses),
    [detailedCalc.totalAccesses, opticalCore]
  );
  const data = useMemo(() => generateData(selectedParam), [
    selectedParam,
    baseParams,
    paramRanges,
    opticalCore
  ]);

  function calculateAccessesDetailed(params) {
    const { embedding_dim: d_model, num_heads: h, sequence_length: L } = params;
    const d_k = d_model / h;

    const steps = [
      {
        name: 'Input Preparation',
        formula: 'X ∈ ℝ^(L×d_model)',
        calculation: `Input tensor: ${L} × ${d_model} = ${L * d_model} elements`,
        accesses: 0,
        description: 'Input patches are already in memory',
        color: '#94a3b8'
      },
      {
        name: 'Query Projection',
        formula: 'Q = X × W_q',
        calculation: `Matrix multiplication: ${L} × ${d_model} × ${d_model} = ${
          L * d_model * d_model
        } FLOPs\nOptical accesses: ${L * d_model}`,
        accesses: L * d_model,
        description: 'Each element of Q requires one optical core access',
        color: '#3b82f6'
      },
      {
        name: 'Key Projection',
        formula: 'K = X × W_k',
        calculation: `Matrix multiplication: ${L} × ${d_model} × ${d_model} = ${
          L * d_model * d_model
        } FLOPs\nOptical accesses: ${L * d_model}`,
        accesses: L * d_model,
        description: 'Each element of K requires one optical core access',
        color: '#10b981'
      },
      {
        name: 'Value Projection',
        formula: 'V = X × W_v',
        calculation: `Matrix multiplication: ${L} × ${d_model} × ${d_model} = ${
          L * d_model * d_model
        } FLOPs\nOptical accesses: ${L * d_model}`,
        accesses: L * d_model,
        description: 'Each element of V requires one optical core access',
        color: '#f59e0b'
      },
      {
        name: 'Attention Scores',
        formula: 'A = (Q × K^T) / √d_k',
        calculation: `For each head: ${L} × ${L} × ${d_k} = ${
          L * L * d_k
        } FLOPs\nAll heads: ${h} × ${L * L * d_k} = ${h * L * L * d_k} FLOPs\nOptical accesses: ${
          L * L * h
        }`,
        accesses: L * L * h,
        description: 'Attention matrix computation - quadratic in sequence length',
        color: '#ef4444'
      },
      {
        name: 'Weighted Sum',
        formula: 'Output = softmax(A) × V',
        calculation: `For each head: ${L} × ${L} × ${d_k} = ${
          L * L * d_k
        } FLOPs\nAll heads: ${h} × ${L * L * d_k} = ${h * L * L * d_k} FLOPs\nOptical accesses: ${
          L * L * h
        }`,
        accesses: L * L * h,
        description: 'Weighted combination using attention weights',
        color: '#8b5cf6'
      }
    ];

    const totalAccesses = steps.reduce((sum, step) => sum + step.accesses, 0);

    return {
      steps,
      totalAccesses,
      projectionAccesses: steps[1].accesses + steps[2].accesses + steps[3].accesses,
      attentionAccesses: steps[4].accesses + steps[5].accesses
    };
  }

  function calculateOpticalMetrics(totalAccesses) {
    const {
      wavelength_channels,
      microrings_per_channel,
      parallel_ops,
      energy_per_access,
      throughput_gops
    } = opticalCore;

    const maxParallelOps = wavelength_channels * microrings_per_channel * parallel_ops;
    const utilizationRatio = totalAccesses / maxParallelOps;
    const energyConsumption = totalAccesses * energy_per_access;
    const executionTime = totalAccesses / (throughput_gops * 1e9);

    return {
      maxParallelOps,
      utilizationRatio,
      energyConsumption,
      executionTime: executionTime * 1000,
      throughputUtilization: ((totalAccesses / 1e9) / throughput_gops) * 100
    };
  }

  function generateData(paramName) {
    const res = [];
    const range = paramRanges[paramName];

    for (let value = range.min; value <= range.max; value += range.step) {
      const params = { ...baseParams };

      if (paramName === 'image_size') {
        params.image_size = value;
        params.sequence_length = Math.pow(value / params.patch_size, 2) + 1;
      } else {
        params[paramName] = value;
      }

      const detailed = calculateAccessesDetailed(params);
      const metrics = calculateOpticalMetrics(detailed.totalAccesses);

      res.push({
        [paramName]: value,
        'Total Accesses': detailed.totalAccesses / 1e6,
        'Projection Accesses': detailed.projectionAccesses / 1e6,
        'Attention Accesses': detailed.attentionAccesses / 1e6,
        'Energy (μJ)': metrics.energyConsumption / 1000,
        'Execution Time (ms)': metrics.executionTime,
        'Utilization (%)': Math.min(metrics.throughputUtilization, 100)
      });
    }
    return res;
  }

  const animateSteps = () => {
    setAnimateCalculation(true);
    setCalculationStep(0);
    const interval = setInterval(() => {
      setCalculationStep(prev => {
        if (prev >= detailedCalc.steps.length - 1) {
          clearInterval(interval);
          setAnimateCalculation(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1500);
  };

  const getXAxisLabel = () => {
    switch (selectedParam) {
      case 'embedding_dim':
        return 'Embedding Dimension';
      case 'sequence_length':
        return 'Sequence Length (# patches + 1)';
      case 'num_heads':
        return 'Number of Attention Heads';
      case 'image_size':
        return 'Image Size (pixels)';
      default:
        return selectedParam;
    }
  };

  const updateNumber = (setter, field, value, min = 1) => {
    const num = Math.max(min, parseInt(value, 10) || min);
    setter(prev => ({ ...prev, [field]: num }));
  };

  const updateRangeNumber = (param, field, value, min = 1) => {
    const num = Math.max(min, parseInt(value, 10) || min);
    setParamRanges(prev => ({
      ...prev,
      [param]: { ...prev[param], [field]: num }
    }));
  };

  const OpticalCoreVisualization = () => {
    const { wavelength_channels, microrings_per_channel } = opticalCore;
    const totalElements = wavelength_channels * microrings_per_channel;

    return (
      <div className="space-y-4">
        <h4 className="font-semibold flex items-center gap-2">
          <Cpu className="w-4 h-4" /> Optical Core Architecture
        </h4>
        <div className="bg-gray-100 p-4 rounded-lg">
          <div className="grid grid-cols-8 gap-1 mb-4">
            {Array.from({ length: Math.min(64, totalElements) }).map((_, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-sm ${
                  i < (detailedCalc.totalAccesses / opticalMetrics.maxParallelOps) * 64
                    ? 'bg-blue-500'
                    : 'bg-gray-300'
                }`}
                title={`Microring ${i + 1}`}
              />
            ))}
          </div>
          <div className="text-xs text-gray-600">
            <div>Wavelength Channels: {wavelength_channels}</div>
            <div>Microrings per Channel: {microrings_per_channel}</div>
            <div>Total Capacity: {totalElements.toLocaleString()} parallel ops</div>
            <div>
              Current Utilization: {(opticalMetrics.utilizationRatio * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    );
  };

  const AttentionPatternVisualization = () => {
    const { sequence_length, num_heads } = baseParams;
    const gridSize = Math.min(12, sequence_length);

    return (
      <div className="space-y-4">
        <h4 className="font-semibold flex items-center gap-2">
          <Eye className="w-4 h-4" /> Attention Pattern ({num_heads} heads)
        </h4>
        <div className="grid grid-cols-4 gap-2">
          {Array.from({ length: Math.min(4, num_heads) }).map((_, headIdx) => (
            <div key={headIdx} className="border rounded p-2">
              <div className="text-xs mb-1">Head {headIdx + 1}</div>
              <div
                className="grid gap-px"
                style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}
              >
                {Array.from({ length: gridSize * gridSize }).map((_, i) => {
                  const intensity = Math.random() * 0.8 + 0.2;
                  return (
                    <div
                      key={i}
                      className="w-2 h-2"
                      style={{
                        backgroundColor: `rgba(59, 130, 246, ${intensity})`
                      }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
        <div className="text-xs text-gray-600">
          Attention matrix: {sequence_length}×{sequence_length} per head
        </div>
      </div>
    );
  };

  const ImagePatchVisualization = () => {
    const { image_size, patch_size, sequence_length } = baseParams;
    const patchesPerSide = image_size / patch_size;

    return (
      <div className="space-y-4">
        <h4 className="font-semibold flex items-center gap-2">
          <Grid className="w-4 h-4" /> Image Patches ({sequence_length - 1} patches + 1 CLS)
        </h4>
        <div className="bg-gray-100 p-4 rounded-lg">
          <div
            className="grid gap-1 mx-auto border-2 border-blue-300"
            style={{
              gridTemplateColumns: `repeat(${patchesPerSide}, 1fr)`,
              width: 'min(200px, 100%)',
              aspectRatio: '1'
            }}
          >
            {Array.from({ length: patchesPerSide * patchesPerSide }).map((_, i) => (
              <div
                key={i}
                className="bg-blue-200 border border-blue-300 flex items-center justify-center text-xs"
                style={{ aspectRatio: '1' }}
              >
                {i + 1}
              </div>
            ))}
          </div>
          <div className="text-xs text-gray-600 mt-2">
            <div>
              Image: {image_size}×{image_size} pixels
            </div>
            <div>
              Patch: {patch_size}×{patch_size} pixels
            </div>
            <div>Grid: {patchesPerSide}×{patchesPerSide} patches</div>
          </div>
        </div>
      </div>
    );
  };

  const pieData = detailedCalc.steps.slice(1).map((step) => ({
    name: step.name,
    value: step.accesses,
    color: step.color
  }));

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Interactive Vision Transformer Optical Core Analysis
        </h1>
        <p className="text-gray-600 mb-6">
          Real-time analysis with dynamic updates and visual representations
        </p>

        <div className="grid lg:grid-cols-4 gap-6 mb-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Settings className="w-5 h-5" /> ViT Parameters
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Embedding Dimension
                </label>
                <input
                  type="number"
                  value={baseParams.embedding_dim}
                  onChange={(e) =>
                    updateNumber(setBaseParams, 'embedding_dim', e.target.value, 64)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="64"
                  step="64"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Number of Heads
                </label>
                <input
                  type="number"
                  value={baseParams.num_heads}
                  onChange={(e) =>
                    updateNumber(setBaseParams, 'num_heads', e.target.value, 1)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Image Size</label>
                <input
                  type="number"
                  value={baseParams.image_size}
                  onChange={(e) =>
                    updateNumber(setBaseParams, 'image_size', e.target.value, 32)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="32"
                  step="32"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Patch Size</label>
                <input
                  type="number"
                  value={baseParams.patch_size}
                  onChange={(e) =>
                    updateNumber(setBaseParams, 'patch_size', e.target.value, 4)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="4"
                />
              </div>
              <div className="text-sm text-gray-600 bg-gray-50 p-2 rounded">
                Sequence Length: <span className="font-mono">{baseParams.sequence_length}</span>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5" /> Optical Core Config
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Wavelength Channels
                </label>
                <input
                  type="number"
                  value={opticalCore.wavelength_channels}
                  onChange={(e) =>
                    updateNumber(setOpticalCore, 'wavelength_channels', e.target.value, 1)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Microrings per Channel
                </label>
                <input
                  type="number"
                  value={opticalCore.microrings_per_channel}
                  onChange={(e) =>
                    updateNumber(setOpticalCore, 'microrings_per_channel', e.target.value, 1)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Parallel Ops</label>
                <input
                  type="number"
                  value={opticalCore.parallel_ops}
                  onChange={(e) =>
                    updateNumber(setOpticalCore, 'parallel_ops', e.target.value, 1)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="1"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">
                  Throughput (GOPS)
                </label>
                <input
                  type="number"
                  value={opticalCore.throughput_gops}
                  onChange={(e) =>
                    updateNumber(setOpticalCore, 'throughput_gops', e.target.value, 1)
                  }
                  className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500"
                  min="1"
                />
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Info className="w-5 h-5" /> Live Metrics
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Total Accesses:</span>
                <span className="font-mono text-blue-600">
                  {(detailedCalc.totalAccesses / 1e6).toFixed(2)}M
                </span>
              </div>
              <div className="flex justify-between">
                <span>Energy:</span>
                <span className="font-mono text-green-600">
                  {(opticalMetrics.energyConsumption / 1000).toFixed(2)}μJ
                </span>
              </div>
              <div className="flex justify-between">
                <span>Execution Time:</span>
                <span className="font-mono text-purple-600">
                  {opticalMetrics.executionTime.toFixed(2)}ms
                </span>
              </div>
              <div className="flex justify-between">
                <span>Utilization:</span>
                <span
                  className={`font-mono ${
                    opticalMetrics.throughputUtilization > 80
                      ? 'text-red-600'
                      : opticalMetrics.throughputUtilization > 60
                      ? 'text-yellow-600'
                      : 'text-green-600'
                  }`}
                >
                  {opticalMetrics.throughputUtilization.toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>Max Parallel:</span>
                <span className="font-mono text-gray-600">
                  {opticalMetrics.maxParallelOps.toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <Calculator className="w-5 h-5" /> Operation Breakdown
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={30}
                  outerRadius={80}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value) => [(value / 1e6).toFixed(2) + 'M', 'Accesses']}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-1 gap-1 text-xs mt-2">
              {pieData.map((entry, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div className="w-3 h-3" style={{ backgroundColor: entry.color }} />
                  <span className="truncate">{entry.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <OpticalCoreVisualization />
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <AttentionPatternVisualization />
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <ImagePatchVisualization />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md mb-6">
          <h3 className="text-lg font-semibold mb-3">Analysis Range Configuration</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(paramRanges).map(([param, range]) => (
              <div key={param} className="space-y-2">
                <label className="block text-sm font-medium capitalize">
                  {param.replace('_', ' ')}
                </label>
                <div className="grid grid-cols-3 gap-1">
                  <input
                    type="number"
                    placeholder="Min"
                    value={range.min}
                    onChange={(e) =>
                      updateRangeNumber(param, 'min', e.target.value, 1)
                    }
                    className="p-2 border rounded text-sm focus:ring-1 focus:ring-blue-500"
                  />
                  <input
                    type="number"
                    placeholder="Max"
                    value={range.max}
                    onChange={(e) =>
                      updateRangeNumber(param, 'max', e.target.value, range.min)
                    }
                    className="p-2 border rounded text-sm focus:ring-1 focus:ring-blue-500"
                  />
                  <input
                    type="number"
                    placeholder="Step"
                    value={range.step}
                    onChange={(e) =>
                      updateRangeNumber(param, 'step', e.target.value, 1)
                    }
                    className="p-2 border rounded text-sm focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow-md mb-6">
          <h3 className="text-lg font-semibold mb-3">Parameter to Analyze:</h3>
          <div className="flex flex-wrap gap-2">
            {[
              { key: 'embedding_dim', label: 'Embedding Dimension', icon: Layers },
              { key: 'sequence_length', label: 'Sequence Length', icon: Grid },
              { key: 'num_heads', label: 'Number of Heads', icon: Eye },
              { key: 'image_size', label: 'Image Size', icon: Grid }
            ].map((param) => {
              const Icon = param.icon;
              return (
                <button
                  key={param.key}
                  onClick={() => setSelectedParam(param.key)}
                  className={`px-4 py-2 rounded-md flex items-center gap-2 transition-all ${
                    selectedParam === param.key
                      ? 'bg-blue-500 text-white shadow-lg'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {param.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h3 className="text-lg font-semibold mb-4">
            Real-time Analysis: {getXAxisLabel()}
            <span className="text-sm font-normal text-gray-500 ml-2">
              (Updates automatically as you change parameters)
            </span>
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey={selectedParam}
                label={{ value: getXAxisLabel(), position: 'insideBottom', offset: -10 }}
              />
              <YAxis
                yAxisId="left"
                label={{
                  value: 'Accesses (Millions)',
                  angle: -90,
                  position: 'insideLeft'
                }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                label={{
                  value: 'Energy (μJ) / Time (ms)',
                  angle: 90,
                  position: 'insideRight'
                }}
              />
              <Tooltip
                formatter={(value, name) => {
                  if (name.includes('Accesses')) return [value.toFixed(2) + 'M', name];
                  if (name.includes('Energy')) return [value.toFixed(2) + 'μJ', name];
                  if (name.includes('Time')) return [value.toFixed(2) + 'ms', name];
                  return [value.toFixed(1) + '%', name];
                }}
              />
              <Legend />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="Total Accesses"
                stroke="#2563eb"
                strokeWidth={3}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="Projection Accesses"
                stroke="#16a34a"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="Attention Accesses"
                stroke="#dc2626"
                strokeWidth={2}
                strokeDasharray="5 5"
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="Energy (μJ)"
                stroke="#f59e0b"
                strokeWidth={2}
              />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="Execution Time (ms)"
                stroke="#8b5cf6"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {showCalculation && (
          <div className="bg-white p-6 rounded-lg shadow-md mb-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Calculator className="w-5 h-5" /> Live Step-by-Step Calculation
              </h3>
              <button
                onClick={animateSteps}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                disabled={animateCalculation}
              >
                <Play className="w-4 h-4" />
                {animateCalculation ? 'Running...' : 'Animate'}
              </button>
            </div>
            <ol className="space-y-3 text-sm">
              {detailedCalc.steps.map((step, index) => (
                <li
                  key={index}
                  className={`p-3 rounded border ${
                    index === calculationStep ? 'bg-blue-50' : 'bg-white'
                  }`}
                >
                  <div className="font-medium mb-1">{step.name}</div>
                  <div className="text-xs font-mono mb-1">{step.formula}</div>
                  {index <= calculationStep && (
                    <div className="text-xs whitespace-pre-wrap text-gray-600">
                      {step.calculation}
                    </div>
                  )}
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
};

export default VisionTransformerAnalysis;