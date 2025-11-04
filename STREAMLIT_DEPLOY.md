# Deploying to Streamlit Cloud

## Quick Start

1. **Fork or clone** this repository to your GitHub account

2. **Get FRED API Key** (free):
   - Visit https://fred.stlouisfed.org/docs/api/api_key.html
   - Register for a free account
   - Copy your API key

3. **Deploy to Streamlit Cloud**:
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Select this repository: `pblo97/MacroArimax`
   - Branch: `claude/liquidity-stress-detection-system-011CUoKdxAbMy1259QPRQkZV` (or `main` after merge)
   - Main file path: `macro_plumbing/app/app.py`
   - Click "Advanced settings"
   - Add secret:
     ```
     FRED_API_KEY = "your_api_key_here"
     ```
   - Click "Deploy"

## Troubleshooting

### Installation Takes Too Long

If installation is hanging:

1. **Use minimal requirements**: Create a `.streamlit/packages.txt` file (empty is fine)

2. **Alternative**: In Streamlit Cloud settings, change the requirements file to use:
   ```
   requirements-minimal.txt
   ```

3. **Check logs**: Look for which package is stuck during installation

### Common Issues

**Issue**: `pandas` building from source (very slow)
- **Solution**: Already fixed in latest `requirements.txt` (no version pinning)
- Uses precompiled wheels automatically

**Issue**: Out of memory during installation
- **Solution**: Use `requirements-minimal.txt` instead
- Remove `scipy` temporarily if needed (some models will be unavailable)

**Issue**: FRED API rate limits
- **Solution**: API key gives 120 requests/minute (plenty for this app)
- Cache is enabled by default (`.fred_cache/` directory)

**Issue**: Import errors for optional packages
- **Solution**: The app is designed to work without optional packages
- Only core functionality is required

## Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export FRED_API_KEY="your_key_here"

# Run app
streamlit run macro_plumbing/app/app.py
```

## File Structure for Deployment

```
MacroArimax/
├── macro_plumbing/
│   ├── app/
│   │   └── app.py          ← Main Streamlit app
│   ├── data/               ← FRED client + config
│   ├── features/           ← Feature engineering
│   ├── models/             ← Statistical models
│   ├── graph/              ← Network analysis
│   └── backtest/           ← Validation
├── requirements.txt        ← Standard (recommended)
├── requirements-minimal.txt ← Minimal (if issues)
└── .streamlit/
    └── secrets.toml        ← Your API key (local only)
```

## Secrets Configuration

Create `.streamlit/secrets.toml` locally (don't commit!):

```toml
# .streamlit/secrets.toml
FRED_API_KEY = "your_api_key_here"
```

On Streamlit Cloud, add the same in the web interface:
- Go to app settings → Secrets
- Paste:
  ```toml
  FRED_API_KEY = "your_api_key_here"
  ```

## Performance Optimization

### Cache Directory

The app uses caching to minimize FRED API calls:
- Cache location: `.fred_cache/`
- Cache duration: 30 days
- Incremental updates: Only fetches new data

### Resource Limits

Streamlit Cloud free tier limits:
- **Memory**: 1 GB RAM
- **CPU**: Shared
- **Storage**: Ephemeral (cache clears on restart)

**Tips**:
- First load will be slower (fetching all data)
- Subsequent loads use cache (much faster)
- Restart app periodically to clear memory

## Testing the Deployment

After deployment:

1. **Wait for installation** (2-3 minutes first time)
2. **Check app loads** (you should see the sidebar)
3. **Enter FRED API key** (if not in secrets)
4. **Click "Run Analysis"**
5. **Verify data loads** (progress bar should show)
6. **Check all 5 tabs** work

## Advanced Configuration

### Custom Python Version

If needed, create `runtime.txt`:
```
python-3.11
```

### System Dependencies

If you need system packages, create `.streamlit/packages.txt`:
```
# Example (usually not needed for this app)
# libgomp1
```

### App Configuration

Create `.streamlit/config.toml` for custom settings:
```toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

## Monitoring

Check app health:
- **Logs**: Available in Streamlit Cloud dashboard
- **Metrics**: Monitor FRED API usage
- **Errors**: Check for import/data errors in logs

## Support

If issues persist:
1. Check the GitHub Issues: https://github.com/pblo97/MacroArimax/issues
2. Review Streamlit docs: https://docs.streamlit.io/streamlit-community-cloud
3. FRED API docs: https://fred.stlouisfed.org/docs/api/

---

**Happy deploying!** 🚀
