# 🚀 Railway Deployment Guide

## Step-by-Step Deployment to Railway

### 1. Prepare Your Repository
- Make sure your code is in a GitHub repository
- Ensure you have the following files:
  - `app.py` ✅
  - `requirements.txt` ✅
  - `Procfile` ✅
  - `runtime.txt` ✅

### 2. Get Your OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (you'll need it in step 4)

### 3. Deploy to Railway
1. **Go to [Railway](https://railway.app)**
2. **Sign up/Login** with your GitHub account
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**
6. **Wait for Railway to detect it's a Python app**

### 4. Configure Environment Variables
1. **Go to your project's "Variables" tab**
2. **Add these environment variables:**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_random_secret_key_here
   ```
3. **Click "Deploy"**

### 5. Get Your Live URL
- Railway will provide you with a URL like: `https://your-app-name.railway.app`
- Share this URL with others to use your app!

### 6. Test Your Deployment
1. **Open your Railway URL**
2. **Allow microphone permissions**
3. **Click "Start Listening"**
4. **Speak into your microphone**
5. **You should see transcriptions and AI suggestions!**

## Troubleshooting

### If deployment fails:
- Check that all files are committed to GitHub
- Verify your OpenAI API key is correct
- Make sure you have billing set up on your OpenAI account

### If audio doesn't work:
- Make sure you're using HTTPS (Railway provides this automatically)
- Check browser microphone permissions
- Try a different browser (Chrome works best)

### If you get API errors:
- Verify your OpenAI API key has credits
- Check the Railway logs for detailed error messages

## Cost Estimation
- **Railway**: Free tier available, then ~$5-10/month
- **OpenAI**: ~$0.01-0.10 per conversation (depending on usage)

## Next Steps
- Customize the UI in `templates/index.html`
- Add more features to `app.py`
- Set up a custom domain (optional)
- Monitor usage in Railway dashboard 