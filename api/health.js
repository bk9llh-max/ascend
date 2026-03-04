export default function handler(req, res) {
  res.status(200).json({
    status: 'healthy',
    timestamp: Date.now(),
    version: '3.0.0',
    environment: process.env.VERCEL_ENV || 'development'
  });
}
