import React, { useState, useEffect } from 'react';
import { 
  Play, Sparkles, Shield, AlertTriangle, CheckCircle, 
  HelpCircle, Eye, Sliders, Database, ArrowRight, Layers, FileCode
} from 'lucide-react';

const POLICY_INFO = {
  WITHHOLD: {
    title: 'WITHHOLD',
    subtitle: 'Nullify ambiguous slots',
    desc: 'Replaces ambiguous or unknown decoded categories with None/null. The safest default for data cleaning.',
    icon: Shield,
    color: 'var(--primary)'
  },
  SENTINEL: {
    title: 'SENTINEL',
    subtitle: 'Informative token injection',
    desc: 'Injects explicit sentinel string <AMBIGUOUS:Category|UNSEEN> into output arrays for immediate alert.',
    icon: Eye,
    color: '#8b5cf6'
  },
  PROVENANCE_SIDE_CHANNEL: {
    title: 'SIDE-CHANNEL',
    subtitle: 'Prediction + audit channel',
    desc: 'Emits candidate reconstruction alongside a structured side-channel metadata dictionary (confidence, collision set).',
    icon: Layers,
    color: '#ec4899'
  },
  STRICT_REJECTION: {
    title: 'STRICT REJECTION',
    subtitle: 'Halt pipeline on ambiguity',
    desc: 'Raises an AmbiguityRejectionError immediately. Designed for safety-critical pipelines (healthcare, credit).',
    icon: AlertTriangle,
    color: 'var(--ambiguous)'
  }
};

export default function SandboxTab({ onOpenProvenance }) {
  const [presets, setPresets] = useState([]);
  const [selectedPresetKey, setSelectedPresetKey] = useState('gender_binary');
  const [presetDesc, setPresetDesc] = useState('');
  
  const [featureNames, setFeatureNames] = useState('Gender');
  const [trainDataText, setTrainDataText] = useState('Female\nMale');
  const [testDataText, setTestDataText] = useState('Female\nMale\nUnknown');
  const [groundTruthText, setGroundTruthText] = useState('Female\nMale\nUnknown');
  
  const [dropConfig, setDropConfig] = useState('if_binary');
  const [handleUnknown, setHandleUnknown] = useState('ignore');
  const [policy, setPolicy] = useState('WITHHOLD');
  
  const [loading, setLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Fetch preset list
  useEffect(() => {
    fetch('/api/presets')
      .then(res => res.json())
      .then(data => {
        setPresets(data.presets || []);
        loadPreset('gender_binary');
      })
      .catch(err => console.error('Presets error', err));
  }, []);

  const loadPreset = async (key) => {
    setSelectedPresetKey(key);
    if (!key) return;
    try {
      const res = await fetch(`/api/preset/${encodeURIComponent(key)}`);
      const { preset } = await res.json();
      if (!preset) return;
      setPresetDesc(preset.description || '');
      setFeatureNames((preset.feature_names || []).join(', '));
      setTrainDataText((preset.train_data || []).map(r => r.join(', ')).join('\n'));
      setTestDataText((preset.test_data || []).map(r => r.join(', ')).join('\n'));
      setGroundTruthText((preset.ground_truth || []).map(r => r.join(', ')).join('\n'));
      setDropConfig(preset.drop === null ? 'null' : preset.drop);
      setHandleUnknown(preset.handle_unknown || 'ignore');
    } catch (e) {
      console.error(e);
    }
  };

  const handleRunAnalysis = async () => {
    setLoading(true);
    setErrorMessage('');

    const train = trainDataText.split('\n').map(l => l.trim()).filter(Boolean).map(l => l.split(',').map(v => v.trim()));
    const test = testDataText.split('\n').map(l => l.trim()).filter(Boolean).map(l => l.split(',').map(v => v.trim()));
    const gt = groundTruthText.split('\n').map(l => l.trim()).filter(Boolean).map(l => l.split(',').map(v => v.trim()));
    const names = featureNames.split(',').map(s => s.trim()).filter(Boolean);

    if (!train.length || !test.length) {
      setErrorMessage('Training and test data are required.');
      setLoading(false);
      return;
    }

    try {
      const payload = {
        train_data: train,
        test_data: test,
        drop: dropConfig === 'null' ? null : dropConfig,
        handle_unknown: handleUnknown,
        policy: policy,
        feature_names: names.length ? names : undefined,
        ground_truth: gt.length ? gt : undefined
      };

      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!data.success) {
        throw new Error(data.error || 'Analysis failed.');
      }
      setAnalysisData(data);
    } catch (err) {
      setErrorMessage(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(340px, 420px) 1fr', gap: '24px', alignItems: 'start' }}>
      
      {/* Left: Configuration Panel */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '18px' }}>
        
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Sliders size={18} style={{ color: 'var(--primary)' }} />
            <h2 style={{ fontSize: '16px', fontWeight: 700 }}>Experiment Setup</h2>
          </div>
          <span className="badge badge-code">Interactive</span>
        </div>

        {/* Preset Selector */}
        <div>
          <label style={{ fontSize: '12.5px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
            Preset Scenario
          </label>
          <select 
            value={selectedPresetKey}
            onChange={(e) => loadPreset(e.target.value)}
            className="select"
          >
            <option value="">— Custom Data —</option>
            {presets.map(p => (
              <option key={p.key} value={p.key}>{p.name}</option>
            ))}
          </select>
          {presetDesc && (
            <div style={{
              fontSize: '12px', color: 'var(--text-secondary)', marginTop: '8px',
              padding: '8px 12px', background: 'var(--bg-surface-elevated)',
              borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--primary)'
            }}>
              {presetDesc}
            </div>
          )}
        </div>

        {/* Feature Names */}
        <div>
          <label style={{ fontSize: '12.5px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
            Feature Names (comma-separated)
          </label>
          <input 
            type="text" 
            value={featureNames}
            onChange={(e) => setFeatureNames(e.target.value)}
            className="input" 
            placeholder="Gender, City, Status"
          />
        </div>

        {/* Strategy Dropdowns */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div>
            <label style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
              drop Strategy
            </label>
            <select 
              value={dropConfig}
              onChange={(e) => setDropConfig(e.target.value)}
              className="select"
            >
              <option value="if_binary">if_binary</option>
              <option value="first">first</option>
              <option value="null">None (No drop)</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '6px' }}>
              handle_unknown
            </label>
            <select 
              value={handleUnknown}
              onChange={(e) => setHandleUnknown(e.target.value)}
              className="select"
            >
              <option value="ignore">ignore (Zeros)</option>
              <option value="error">error (Raise)</option>
            </select>
          </div>
        </div>

        {/* Ambiguity Resolution Policy Selector (Deliverable D3) */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <label style={{ fontSize: '12.5px', fontWeight: 700 }}>
              Ambiguity Resolution Policy
            </label>
            <span className="badge badge-code" style={{ fontSize: '10px' }}>Safety Policies</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {Object.entries(POLICY_INFO).map(([key, info]) => {
              const Icon = info.icon;
              const isSelected = policy === key;
              return (
                <div
                  key={key}
                  onClick={() => setPolicy(key)}
                  style={{
                    padding: '10px',
                    borderRadius: 'var(--radius-sm)',
                    border: `1.5px solid ${isSelected ? info.color : 'var(--border-subtle)'}`,
                    background: isSelected ? 'var(--bg-surface-subtle)' : 'var(--bg-surface-elevated)',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    boxShadow: isSelected ? `0 0 10px ${info.color}33` : 'none'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                    <Icon size={14} style={{ color: info.color }} />
                    <span style={{ fontSize: '11.5px', fontWeight: 700, color: isSelected ? info.color : 'var(--text-primary)' }}>
                      {info.title}
                    </span>
                  </div>
                  <div style={{ fontSize: '10.5px', color: 'var(--text-muted)', lineHeight: 1.2 }}>
                    {info.subtitle}
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)', marginTop: '8px', fontStyle: 'italic' }}>
            {POLICY_INFO[policy].desc}
          </div>
        </div>

        {/* Data Inputs */}
        <div>
          <label style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
            Training Data (Categories to fit)
          </label>
          <textarea 
            rows={3} 
            value={trainDataText}
            onChange={(e) => setTrainDataText(e.target.value)}
            className="textarea" 
            placeholder="Female&#10;Male"
          />
        </div>

        <div>
          <label style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
            Test Data (Include novel/unseen values)
          </label>
          <textarea 
            rows={3} 
            value={testDataText}
            onChange={(e) => setTestDataText(e.target.value)}
            className="textarea" 
            placeholder="Female&#10;Male&#10;Unknown"
          />
        </div>

        <div>
          <label style={{ fontSize: '12px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
            Ground Truth (Optional, for correctness scoring)
          </label>
          <textarea 
            rows={3} 
            value={groundTruthText}
            onChange={(e) => setGroundTruthText(e.target.value)}
            className="textarea" 
            placeholder="Female&#10;Male&#10;Unknown"
          />
        </div>

        {errorMessage && (
          <div style={{
            padding: '10px 14px', borderRadius: 'var(--radius-sm)',
            background: 'var(--ambiguous-tint)', border: '1px solid var(--ambiguous)',
            color: 'var(--ambiguous)', fontSize: '12.5px'
          }}>
            {errorMessage}
          </div>
        )}

        <button 
          onClick={handleRunAnalysis}
          disabled={loading}
          className="btn btn-primary"
          style={{ width: '100%', padding: '12px', fontSize: '14px', marginTop: '4px' }}
        >
          {loading ? <span className="spinner" /> : <Play size={16} />}
          <span>{loading ? 'Analyzing Vector Collisions…' : 'Run Safe Decoding Analysis'}</span>
        </button>

      </div>

      {/* Right: Results Display */}
      <div>
        {!analysisData ? (
          <div className="glass-panel" style={{ padding: '60px 30px', textAlign: 'center' }}>
            <div style={{
              width: '64px', height: '64px', borderRadius: '50%',
              background: 'var(--primary-tint)', color: 'var(--primary)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto 18px auto'
            }}>
              <Sparkles size={32} />
            </div>
            <h3 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '8px' }}>
              Ready for Inverse Decoding Analysis
            </h3>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '480px', margin: '0 auto 20px auto', fontSize: '13.5px' }}>
              Select a preset scenario on the left or customize your training data, pick your preferred Ambiguity Resolution Policy, and click <strong>Run Safe Decoding Analysis</strong>.
            </p>
            <button onClick={handleRunAnalysis} className="btn btn-secondary">
              <Play size={14} /> Run Canonical Gender Preset
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* Key Finding Banner */}
            <div className="glass-panel" style={{
              padding: '16px 20px',
              borderLeft: '4px solid var(--primary)',
              background: 'linear-gradient(90deg, var(--primary-tint) 0%, var(--bg-surface) 100%)',
              display: 'flex', alignItems: 'flex-start', gap: '12px'
            }}>
              <Sparkles size={20} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: '2px' }} />
              <div>
                <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--primary)', letterSpacing: '0.05em' }}>
                  Core Research Finding
                </div>
                <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', marginTop: '2px', lineHeight: 1.5 }}>
                  {analysisData.summary.key_finding}
                </div>
              </div>
            </div>

            {/* Policy Rejection Alert (If STRICT_REJECTION triggered) */}
            {analysisData.rejected && (
              <div className="glass-panel" style={{
                padding: '16px 20px',
                borderLeft: '4px solid var(--ambiguous)',
                background: 'var(--ambiguous-tint)',
                display: 'flex', alignItems: 'flex-start', gap: '12px'
              }}>
                <AlertTriangle size={22} style={{ color: 'var(--ambiguous)', flexShrink: 0 }} />
                <div>
                  <h4 style={{ fontSize: '15px', color: 'var(--ambiguous)', fontWeight: 700, margin: '0 0 4px 0' }}>
                    Pipeline Halted by STRICT_REJECTION Policy
                  </h4>
                  <p style={{ fontSize: '13px', margin: 0, color: 'var(--text-primary)' }}>
                    {analysisData.rejection.message} &middot; Collision candidates: <strong>{analysisData.rejection.possible_values.join(', ')}</strong>
                  </p>
                </div>
              </div>
            )}

            {/* Metric Scorecards */}
            {!analysisData.rejected && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px' }}>
                
                <div className="glass-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Silent Errors Blocked
                  </div>
                  <div style={{ fontSize: '28px', fontWeight: 800, color: analysisData.metrics.baseline_incorrect > 0 ? 'var(--ambiguous)' : 'var(--safe)', margin: '4px 0' }}>
                    {analysisData.metrics.baseline_incorrect}
                  </div>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                    sklearn misclassifications caught
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Ambiguity Recall
                  </div>
                  <div style={{ fontSize: '28px', fontWeight: 800, color: 'var(--safe)', margin: '4px 0' }}>
                    {analysisData.metrics.detection_rate}%
                  </div>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                    {analysisData.metrics.ambiguous_count} collisions trapped
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Safe Handling Rate
                  </div>
                  <div style={{ fontSize: '28px', fontWeight: 800, color: 'var(--primary)', margin: '4px 0' }}>
                    {analysisData.metrics.safe_handling_rate}%
                  </div>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                    Zero corrupted reconstructions
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                    Baseline Accuracy
                  </div>
                  <div style={{ fontSize: '28px', fontWeight: 800, color: 'var(--text-secondary)', margin: '4px 0' }}>
                    {analysisData.summary.baseline_accuracy}
                  </div>
                  <div style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>
                    Raw scikit-learn performance
                  </div>
                </div>

              </div>
            )}

            {/* Collision Breakdown Diagram */}
            <div className="glass-panel" style={{ padding: '18px 22px' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={16} style={{ color: 'var(--primary)' }} />
                <span>One-Hot Sub-Vector Collision Mechanism (scikit-learn Issue #34549)</span>
              </div>
              
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr auto 1fr auto 1fr',
                alignItems: 'center', gap: '12px', padding: '16px',
                background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)',
                fontSize: '12.5px'
              }}>
                <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>Dropped Known</div>
                  <strong style={{ fontSize: '13.5px', color: 'var(--text-primary)' }}>Female</strong>
                  <div style={{ fontSize: '11px', color: 'var(--primary-light)' }}>Encodes to [0]</div>
                </div>

                <div style={{ textAlign: 'center', color: 'var(--ambiguous)', fontWeight: 800, fontSize: '16px' }}>+</div>

                <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
                  <div style={{ color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase' }}>Unseen Category</div>
                  <strong style={{ fontSize: '13.5px', color: 'var(--text-primary)' }}>Unknown</strong>
                  <div style={{ fontSize: '11px', color: 'var(--unknown)' }}>Encodes to [0]</div>
                </div>

                <div style={{ textAlign: 'center', color: 'var(--primary)', fontWeight: 800 }}>&rarr;</div>

                <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-sm)', background: 'var(--ambiguous-tint)', border: '1px solid var(--ambiguous)', textAlign: 'center' }}>
                  <div style={{ color: 'var(--ambiguous)', fontSize: '11px', textTransform: 'uppercase', fontWeight: 700 }}>Collision Vector</div>
                  <strong className="code-pill" style={{ fontSize: '14px', display: 'inline-block', margin: '3px 0' }}>[0]</strong>
                  <div style={{ fontSize: '11px', color: 'var(--ambiguous)' }}>sklearn blindly returns "Female"</div>
                </div>
              </div>
            </div>

            {/* Side-by-Side Comparison Table */}
            {!analysisData.rejected && (
              <div className="glass-panel" style={{ padding: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '14px' }}>
                  <div>
                    <h3 style={{ fontSize: '16px', fontWeight: 700 }}>Side-by-Side Reconstruction Comparison</h3>
                    <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)' }}>
                      Compare vanilla scikit-learn output against the Ambiguity-Safe layer under policy <span className="badge badge-code">{analysisData.config.policy}</span>
                    </p>
                  </div>
                </div>

                <div className="table-container">
                  <table className="modern-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Input</th>
                        <th>Encoded Vector</th>
                        <th>Baseline (sklearn)</th>
                        <th>Status</th>
                        <th>Safe Output</th>
                        <th>Candidate Set</th>
                        <th>Audit Trace</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysisData.results.map((row, idx) => {
                        const isSilent = row.baseline_correct === false;
                        const isAmb = row.row_status === 'AMBIGUOUS';
                        return (
                          <tr key={idx}>
                            <td>{row.sample_index + 1}</td>
                            <td><strong style={{ fontFamily: 'var(--font-mono)' }}>{(row.input || []).join(', ')}</strong></td>
                            <td><span className="code-pill">[{row.encoded.join(', ')}]</span></td>
                            <td>
                              <span className={isSilent ? 'silent-error-highlight' : ''}>
                                {row.baseline_decoded.join(', ')} {isSilent ? '⚠️ (SILENT FAILURE)' : ''}
                              </span>
                            </td>
                            <td>
                              <span className={`badge badge-${row.row_status.toLowerCase()}`}>
                                {row.row_status}
                              </span>
                            </td>
                            <td>
                              <strong style={{ color: row.safe_output.includes(null) ? 'var(--text-muted)' : 'var(--text-primary)' }}>
                                {row.safe_output.map(v => v === null ? 'None' : v).join(', ')}
                              </strong>
                            </td>
                            <td>
                              <span className="code-pill">
                                {row.possible_values.map(p => `[${p.join('|')}]`).join(', ')}
                              </span>
                            </td>
                            <td>
                              <button 
                                onClick={() => onOpenProvenance(row)}
                                className="btn btn-secondary btn-sm"
                                style={{ gap: '4px' }}
                              >
                                <FileCode size={13} /> Inspect
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Encoder Metadata & Collision Boundaries */}
            <div className="glass-panel" style={{ padding: '20px' }}>
              <div style={{ fontSize: '15px', fontWeight: 700, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Database size={16} style={{ color: 'var(--primary)' }} />
                <span>Encoder Metadata & Collision Boundaries</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px' }}>
                {analysisData.metadata.features.map(f => (
                  <div key={f.feature_index} style={{
                    padding: '14px', borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-surface-elevated)', border: '1px solid var(--border-subtle)'
                  }}>
                    <h4 style={{ fontSize: '14px', color: 'var(--primary-light)', marginBottom: '8px' }}>
                      {f.feature_name} (Column {f.feature_index})
                    </h4>
                    <div style={{ fontSize: '12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                      <div>Categories: <strong>{f.categories.join(', ')}</strong></div>
                      <div>Dropped: <strong>{f.dropped_category || 'None'}</strong></div>
                      <div>
                        Collision Hazard: 
                        <span className={`badge ${f.can_produce_all_zeros ? 'badge-ambiguous' : 'badge-safe'}`} style={{ marginLeft: '6px' }}>
                          {f.can_produce_all_zeros ? 'HAZARDOUS (All-Zeros Collision)' : 'SAFE'}
                        </span>
                      </div>
                      <div>Encoded Column Span: <span className="code-pill">{f.encoded_col_range.join('..')}</span> ({f.n_encoded_columns} cols)</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        )}
      </div>

    </div>
  );
}
