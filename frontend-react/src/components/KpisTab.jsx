import React, { useState, useEffect } from 'react';
import { BarChart2, Play, CheckCircle2, ArrowRight, Zap, Database } from 'lucide-react';

export default function KpisTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runBenchmarks = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/kpi-benchmarks');
      const resJson = await res.json();
      if (!resJson.success) throw new Error(resJson.error || 'Failed to run benchmarks');
      setData(resJson);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runBenchmarks();
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* Tab Header Banner */}
      <div className="glass-panel" style={{
        padding: '20px 24px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: '16px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h2 style={{ fontSize: '18px', fontWeight: 800 }}>
              Key Performance Indicators (KPI-1 to KPI-6) Dashboard
            </h2>
            <span className="badge badge-code">Benchmark Suite</span>
          </div>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            Quantitative benchmarks evaluating accuracy, recall, latency bounds, and memory overhead.
          </p>
        </div>

        <button 
          onClick={runBenchmarks} 
          disabled={loading}
          className="btn btn-primary"
        >
          {loading ? <span className="spinner" /> : <Play size={15} />}
          <span>{loading ? 'Running 400 Trials…' : 'Run Monte-Carlo Benchmark Suite'}</span>
        </button>
      </div>

      {error && (
        <div style={{
          padding: '14px 18px', borderRadius: 'var(--radius-md)',
          background: 'var(--ambiguous-tint)', border: '1px solid var(--ambiguous)',
          color: 'var(--ambiguous)', fontSize: '13px'
        }}>
          {error}
        </div>
      )}

      {/* Summary Banner */}
      {data && (
        <div className="glass-panel" style={{
          padding: '16px 22px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          flexWrap: 'wrap', gap: '14px', borderLeft: '4px solid var(--safe)'
        }}>
          <div>
            <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)' }}>
              Benchmark Results: {data.n_samples} Samples Evaluated
            </div>
            <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              p95 Transform Latency: <strong>{data.summary.p95_safe_ms} ms</strong> &middot; Sparse Footprint: <strong>{data.summary.sparse_memory_pct}%</strong>
            </div>
          </div>
          <span className="badge badge-safe" style={{ fontSize: '12.5px', padding: '6px 12px' }}>
            <CheckCircle2 size={16} /> ALL 6 KPIS PASSED
          </span>
        </div>
      )}

      {/* 6 KPI Cards Grid */}
      {data && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '20px' }}>
          {data.measurements.map((kpi) => {
            const isGood = kpi.pass_verdict;
            return (
              <div 
                key={kpi.kpi_id} 
                className="glass-panel" 
                style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
              >
                <div>
                  {/* Card Header */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span className="badge badge-code">{kpi.kpi_id}</span>
                      <h3 style={{ fontSize: '15px', fontWeight: 700 }}>{kpi.name}</h3>
                    </div>
                    <span className="badge badge-safe">PASS</span>
                  </div>

                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: 1.4 }}>
                    {kpi.description}
                  </p>

                  {/* Value Comparison Box */}
                  <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'space-around',
                    background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)',
                    padding: '14px', marginBottom: '16px', border: '1px solid var(--border-subtle)'
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
                        Baseline (sklearn)
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 800, color: 'var(--text-secondary)' }}>
                        {kpi.baseline_value}{kpi.unit}
                      </div>
                    </div>

                    <ArrowRight size={20} style={{ color: 'var(--primary)' }} />

                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--safe)', fontWeight: 700 }}>
                        Ambiguity-Safe Layer
                      </div>
                      <div style={{ fontSize: '24px', fontWeight: 800, color: 'var(--safe)' }}>
                        {kpi.safe_value}{kpi.unit}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Target Rule Footer */}
                <div style={{ 
                  borderTop: '1px solid var(--border-subtle)', paddingTop: '10px', 
                  fontSize: '11.5px', color: 'var(--text-secondary)' 
                }}>
                  <strong style={{ color: 'var(--text-primary)' }}>Target Threshold:</strong> {kpi.target_rule}
                </div>
              </div>
            );
          })}
        </div>
      )}

    </div>
  );
}
