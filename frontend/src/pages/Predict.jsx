import { useState } from 'react'
import { 
  ScanSearch, Shuffle, AlertTriangle, ShieldCheck, ShieldAlert,
  ChevronDown, ChevronUp, Zap, Send, RotateCcw, Sparkles
} from 'lucide-react'
import RiskBadge from '../components/RiskBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { predict } from '../services/api'

// Generate random transaction values
function generateRandom() {
  const values = { Amount: parseFloat((Math.random() * 500).toFixed(2)) }
  for (let i = 1; i <= 28; i++) {
    values[`V${i}`] = parseFloat((Math.random() * 6 - 3).toFixed(4))
  }
  return values
}

// Generate fraud-like values (abnormal V patterns)
function generateFraudLike() {
  const values = { Amount: parseFloat((Math.random() * 2000 + 500).toFixed(2)) }
  const fraudPatterns = {
    V1: -3.5, V2: 2.8, V3: -4.2, V4: 3.1, V5: -2.7,
    V6: -1.9, V7: -5.1, V8: 1.2, V9: -3.8, V10: -6.2,
    V11: 3.4, V12: -7.1, V13: 0.5, V14: -8.3, V15: 0.2,
    V16: -5.5, V17: -6.8, V18: -2.1, V19: 1.7, V20: 0.9,
    V21: 0.8, V22: 0.3, V23: -0.4, V24: -0.1, V25: 0.4,
    V26: -0.2, V27: 0.7, V28: 0.3,
  }
  for (let i = 1; i <= 28; i++) {
    const base = fraudPatterns[`V${i}`]
    values[`V${i}`] = parseFloat((base + (Math.random() - 0.5) * 1.5).toFixed(4))
  }
  return values
}

function ProbabilityGauge({ probability }) {
  const pct = (probability * 100)
  const color = pct > 70 ? '#ef4444' : pct > 40 ? '#f59e0b' : '#10b981'

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-xs font-medium text-slate-400">Fraud Probability</span>
        <span className="text-lg font-bold" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-3 bg-slate-700/60 rounded-full overflow-hidden">
        <div 
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{ 
            width: `${pct}%`, 
            background: `linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%)`,
          }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-slate-500">
        <span>Safe</span>
        <span>Suspicious</span>
        <span>Fraud</span>
      </div>
    </div>
  )
}

function Predict() {
  const [formData, setFormData] = useState(generateRandom())
  const [showFeatures, setShowFeatures] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value === '' ? '' : parseFloat(value) || 0 }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const payload = {}
      payload.Amount = Number(formData.Amount) || 0
      for (let i = 1; i <= 28; i++) {
        payload[`V${i}`] = Number(formData[`V${i}`]) || 0
      }
      const data = await predict(payload)
      setResult(data)
    } catch (err) {
      // Fallback demo result
      const prob = Math.random()
      setResult({
        prediction: prob > 0.5 ? 'Fraud' : 'Legit',
        probability: prob,
        risk_level: prob > 0.7 ? 'high' : prob > 0.4 ? 'medium' : 'low',
        confidence: (1 - Math.abs(prob - 0.5) * 2) * 100,
      })
      setError('Backend unavailable — showing simulated result')
    } finally {
      setLoading(false)
    }
  }

  const fillRandom = () => {
    setFormData(generateRandom())
    setResult(null)
  }

  const fillFraud = () => {
    setFormData(generateFraudLike())
    setResult(null)
  }

  const resetForm = () => {
    const blank = { Amount: 0 }
    for (let i = 1; i <= 28; i++) blank[`V${i}`] = 0
    setFormData(blank)
    setResult(null)
    setError(null)
  }

  const isFraud = result?.prediction === 'Fraud'

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-white tracking-tight">Predict Transaction</h1>
        <p className="text-slate-400 mt-1">Analyze a credit card transaction for potential fraud</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-3 space-y-5">
          {/* Amount Card */}
          <div className="glass-card p-6">
            <label className="block text-sm font-semibold text-white mb-3">
              Transaction Amount ($)
            </label>
            <div className="relative">
              <span className="absolute left-4 top-1/2 -translate-y-1/2 text-emerald-400 font-bold text-lg">$</span>
              <input
                type="number"
                step="0.01"
                value={formData.Amount}
                onChange={(e) => handleChange('Amount', e.target.value)}
                className="input-field pl-10 text-xl font-semibold"
                placeholder="0.00"
              />
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex flex-wrap gap-3">
            <button type="button" onClick={fillRandom} className="btn-secondary flex items-center gap-2 text-sm">
              <Shuffle className="w-4 h-4 text-blue-400" />
              Random Transaction
            </button>
            <button type="button" onClick={fillFraud} className="btn-secondary flex items-center gap-2 text-sm">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              Fraud-like Transaction
            </button>
            <button type="button" onClick={resetForm} className="btn-secondary flex items-center gap-2 text-sm">
              <RotateCcw className="w-4 h-4 text-slate-400" />
              Reset
            </button>
          </div>

          {/* V Features Collapsible */}
          <div className="glass-card overflow-hidden">
            <button
              type="button"
              onClick={() => setShowFeatures(!showFeatures)}
              className="w-full flex items-center justify-between p-5 text-left hover:bg-slate-800/30 transition-colors duration-200"
            >
              <div className="flex items-center gap-3">
                <Sparkles className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="text-sm font-semibold text-white">PCA Features (V1-V28)</p>
                  <p className="text-xs text-slate-500">Principal component analysis transformed features</p>
                </div>
              </div>
              {showFeatures ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
            </button>

            {showFeatures && (
              <div className="px-5 pb-5 animate-fade-in">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {Array.from({ length: 28 }, (_, i) => i + 1).map(i => (
                    <div key={i}>
                      <label className="block text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-1">
                        V{i}
                      </label>
                      <input
                        type="number"
                        step="0.0001"
                        value={formData[`V${i}`]}
                        onChange={(e) => handleChange(`V${i}`, e.target.value)}
                        className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/30 rounded-lg text-sm text-slate-300 focus:outline-none focus:border-emerald-500/50 transition-colors"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={loading}
            className="btn-primary w-full flex items-center justify-center gap-2 text-base py-4"
          >
            {loading ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing Transaction...
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Analyze Transaction
              </>
            )}
          </button>
        </form>

        {/* Result Panel */}
        <div className="lg:col-span-2 space-y-5">
          {/* Awaiting state */}
          {!result && !loading && (
            <div className="glass-card p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-slate-800/60 flex items-center justify-center">
                <ScanSearch className="w-8 h-8 text-slate-500" />
              </div>
              <h3 className="text-lg font-semibold text-slate-300 mb-2">Ready to Analyze</h3>
              <p className="text-sm text-slate-500 leading-relaxed">
                Fill in the transaction details and click "Analyze Transaction" to get a fraud prediction.
              </p>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="glass-card p-8">
              <LoadingSpinner text="Running ML models..." />
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl px-4 py-3 flex items-center gap-3 animate-slide-up">
              <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
              <p className="text-xs text-amber-300">{error}</p>
            </div>
          )}

          {/* Result Card */}
          {result && !loading && (
            <div className={`glass-card overflow-hidden animate-scale-in ${isFraud ? 'glow-red border-red-500/20' : 'glow-emerald border-emerald-500/20'}`}>
              {/* Header Banner */}
              <div className={`px-6 py-5 ${
                isFraud 
                  ? 'bg-gradient-to-r from-red-500/10 to-red-900/5' 
                  : 'bg-gradient-to-r from-emerald-500/10 to-emerald-900/5'
              }`}>
                <div className="flex items-center gap-3">
                  <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                    isFraud ? 'bg-red-500/20' : 'bg-emerald-500/20'
                  }`}>
                    {isFraud ? (
                      <ShieldAlert className="w-6 h-6 text-red-400" />
                    ) : (
                      <ShieldCheck className="w-6 h-6 text-emerald-400" />
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">Prediction Result</p>
                    <p className={`text-2xl font-bold ${isFraud ? 'text-red-400' : 'text-emerald-400'}`}>
                      {isFraud ? 'Fraudulent' : 'Legitimate'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Details */}
              <div className="p-6 space-y-5">
                <ProbabilityGauge probability={result.probability || 0} />

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-900/40 rounded-xl p-4 text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Risk Level</p>
                    <RiskBadge level={result.risk_level || (result.probability > 0.7 ? 'high' : result.probability > 0.4 ? 'medium' : 'low')} />
                  </div>
                  <div className="bg-slate-900/40 rounded-xl p-4 text-center">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Amount</p>
                    <p className="text-lg font-bold text-white">${Number(formData.Amount).toFixed(2)}</p>
                  </div>
                </div>

                {result.confidence && (
                  <div className="bg-slate-900/40 rounded-xl p-4">
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Model Confidence</p>
                    <p className="text-lg font-bold text-blue-400">{Number(result.confidence).toFixed(1)}%</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Info */}
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Zap className="w-4 h-4 text-amber-400" />
              <h4 className="text-sm font-semibold text-white">How it works</h4>
            </div>
            <ul className="space-y-2 text-xs text-slate-400">
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                V1-V28 are PCA-transformed features from original transaction data
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 flex-shrink-0" />
                Our ensemble model (XGBoost + RF) analyzes feature patterns
              </li>
              <li className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 mt-1.5 flex-shrink-0" />
                Probability score indicates likelihood of fraudulent activity
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Predict
