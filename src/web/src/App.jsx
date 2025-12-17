import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Wheat, AlertCircle, CheckCircle, Activity, ScanLine } from 'lucide-react';
import axios from 'axios';
import DropZone from './components/DropZone';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleDrop = async (acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    setFile(selectedFile);
    setPrediction(null);
    setError(null);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Point to Backend (Env var for Prod, localhost for Dev)
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setPrediction(response.data);
    } catch (err) {
      setError("Analysis failed. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-cyber-black text-white p-6 relative overflow-hidden font-sans selection:bg-cyber-green selection:text-black">
      {/* Background Gradients */}
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-cyber-green/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-blue-500/10 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-4xl mx-auto relative z-10">
        {/* Header */}
        <header className="flex items-center justify-between mb-12">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyber-green/20 rounded-lg border border-cyber-green/50">
              <Wheat className="w-8 h-8 text-cyber-green" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Wheat<span className="text-cyber-green">Guard</span> AI</h1>
              <p className="text-white/40 text-sm">Flagship Disease Detection System</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-cyber-green/70 bg-cyber-green/5 px-3 py-1 rounded-full border border-cyber-green/20">
            <Activity className="w-3 h-3" />
            SYSTEM ONLINE
          </div>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">

          {/* Left Column: Upload */}
          <section className="glass-panel p-6 min-h-[400px] flex flex-col items-center justify-center relative group">
            <DropZone onFileAccepted={handleDrop} file={file} />

            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="absolute inset-0 bg-cyber-black/80 backdrop-blur-sm z-20 flex flex-col items-center justify-center rounded-2xl"
              >
                <div className="relative">
                  <div className="w-16 h-16 border-4 border-cyber-green/20 border-t-cyber-green rounded-full animate-spin" />
                  <ScanLine className="absolute inset-0 m-auto text-cyber-green w-6 h-6 animate-pulse" />
                </div>
                <p className="mt-4 font-mono text-cyber-green animate-pulse">ANALYZING LEAF STRUCTURE...</p>
              </motion.div>
            )}
          </section>

          {/* Right Column: Results */}
          <section className="space-y-6">
            <AnimatePresence mode="wait">
              {!prediction && !error && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="glass-panel p-8 text-center h-[400px] flex flex-col items-center justify-center text-white/30"
                >
                  <ScanLine className="w-16 h-16 mb-4 opacity-50" />
                  <p>Awaiting satellite imagery or upload...</p>
                </motion.div>
              )}

              {error && (
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="glass-panel p-6 border-red-500/30 bg-red-500/5 text-red-500 flex items-center gap-3"
                >
                  <AlertCircle />
                  {error}
                </motion.div>
              )}

              {prediction && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="glass-panel p-0 overflow-hidden"
                >
                  <div className="p-6 border-b border-white/10 bg-white/5">
                    <h2 className="text-sm font-mono text-white/50 uppercase mb-1">Diagnosis</h2>
                    <div className="flex items-center gap-3">
                      <h3 className="text-3xl font-bold text-white">{prediction.prediction}</h3>
                      {prediction.prediction === 'Healthy Wheat' ? (
                        <CheckCircle className="text-green-500 w-8 h-8" />
                      ) : prediction.prediction.startsWith('Unknown') ? (
                        <Activity className="text-yellow-500 w-8 h-8" />
                      ) : (
                        <AlertCircle className="text-red-500 w-8 h-8" />
                      )}
                    </div>
                  </div>

                  <div className="p-6 space-y-4">
                    {/* Alert Message for Low Confidence */}
                    {prediction.alert && (
                      <div className="bg-yellow-500/10 border border-yellow-500/20 text-yellow-500 p-3 rounded-lg text-sm flex items-start gap-2 mb-4">
                        <AlertCircle className="w-5 h-5 shrink-0" />
                        <p>{prediction.alert}</p>
                      </div>
                    )}
                    <div className="flex justify-between items-end mb-2">
                      <span className="text-white/60">Confidence Score</span>
                      <span className={`text-2xl font-mono ${prediction.prediction.startsWith('Unknown') ? 'text-yellow-500' : 'text-cyber-green'}`}>{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                    {/* Confidence Bar */}
                    <div className="h-4 bg-black/50 rounded-full overflow-hidden border border-white/10 relative">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${prediction.confidence * 100}%` }}
                        transition={{ duration: 1, ease: "circOut" }}
                        className={`h-full shadow-[0_0_20px_rgba(0,255,157,0.5)] ${prediction.prediction.startsWith('Unknown') ? 'bg-yellow-500' : 'bg-gradient-to-r from-cyber-green to-blue-500'}`}
                      />
                    </div>

                    <div className="pt-4 border-t border-white/10 mt-6 grid grid-cols-2 gap-2 text-xs font-mono text-white/40">
                      {Object.entries(prediction.scores).map(([key, val]) => (
                        <div key={key} className="flex justify-between">
                          <span>{key.split(' ').pop()}</span>
                          <span>{(val * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>

        </div>
      </div>
    </div>
  );
}

export default App;
