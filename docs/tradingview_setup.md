# TradingView Setup Guide

This guide will help you configure TradingView alerts to send data to the NQ Signal Bot webhook server.

## Prerequisites

1. **TradingView Account**: You need a TradingView account (free or paid)
2. **NQ Symbol Access**: Ensure you have access to NQ1! or NQ futures symbol
3. **Webhook Server Running**: Your NQ Signal Bot webhook server must be running and accessible

## Step 1: Create the Alert

1. Open TradingView and go to the NQ1! chart
2. Click on the "Alert" button (bell icon) or press `Alt + A`
3. Configure the alert settings:

### Alert Settings
- **Condition**: Choose any condition that will trigger frequently
  - For testing: "Close" "Crossing" "Previous close"
  - For live trading: Choose technical indicators that match your strategy
- **Options**: 
  - Expiration: Set to "Open-ended" for continuous operation
  - Alert name: "NQ Signal Feed" or similar

### Webhook Settings
1. Check the "Webhook URL" checkbox
2. Enter your webhook URL:
   ```
   http://your-server-ip:8000/webhook
   ```
   - Replace `your-server-ip` with your server's IP address or domain
   - Use `localhost:8000` if running locally

### Message Configuration
In the "Message" field, enter the following JSON:

```json
{
  "timestamp": "{{timenow}}",
  "symbol": "{{ticker}}",
  "timeframe": "{{interval}}",
  "open": {{open}},
  "high": {{high}},
  "low": {{low}},
  "close": {{close}},
  "volume": {{volume}}
}
```

### Important Notes:
- The double curly braces `{{}}` are TradingView placeholders
- These will be replaced with actual values when the alert triggers
- Ensure the JSON is valid (no trailing commas)

## Step 2: Test the Alert

1. Click "Create" to save the alert
2. The alert should appear in your alerts list
3. Wait for the alert to trigger or use "Test Alert" if available

### Verification Steps:
1. Check your webhook server logs for incoming requests
2. Verify the JSON payload is being received correctly
3. Check the database for new market data entries
4. Look for signal generation in the logs

## Step 3: Multiple Timeframes (Optional)

For best results, set up alerts on multiple timeframes:

### Recommended Setup:
1. **1-minute chart**: For precise entries
2. **5-minute chart**: Primary analysis timeframe
3. **15-minute chart**: HTF confirmation
4. **1-hour chart**: Market structure

### Alert Configuration for Each Timeframe:
```json
{
  "timestamp": "{{timenow}}",
  "symbol": "{{ticker}}",
  "timeframe": "1",  // Change to "5", "15", "60" for other timeframes
  "open": {{open}},
  "high": {{high}},
  "low": {{low}},
  "close": {{close}},
  "volume": {{volume}}
}
```

## Step 4: Advanced Alert Conditions

### For Better Signal Quality:
Instead of simple price crossing, use technical indicators:

**Example - RSI Alert:**
```json
{
  "timestamp": "{{timenow}}",
  "symbol": "{{ticker}}",
  "timeframe": "{{interval}}",
  "open": {{open}},
  "high": {{high}},
  "low": {{low}},
  "close": {{close}},
  "volume": {{volume}}
}
```

**Alert Condition**: RSI crosses above 30 or below 70

### For Volume Confirmation:
Add volume conditions to your alerts to ensure sufficient market activity.

## Troubleshooting

### Alert Not Triggering:
1. Check if the condition is met on the chart
2. Ensure the alert is enabled (not paused)
3. Verify the symbol is active and trading

### Webhook Not Receiving Data:
1. Check server logs for incoming requests
2. Verify the webhook URL is correct and accessible
3. Test with a simple curl command:
   ```bash
   curl -X POST http://your-server:8000/webhook \
     -H "Content-Type: application/json" \
     -d '{"test": "data"}'
   ```

### JSON Parse Errors:
1. Ensure all placeholders are correctly formatted
2. Check for missing commas or brackets
3. Validate the JSON structure

### Database Issues:
1. Verify database file permissions
2. Check disk space
3. Review database logs

## Best Practices

### Alert Frequency:
- **Development**: Every 5-15 minutes for testing
- **Production**: Every 1-5 minutes depending on strategy
- **Avoid**: Sub-minute intervals to prevent over-analysis

### Server Setup:
1. Use a stable internet connection
2. Consider a VPS for 24/7 operation
3. Set up monitoring and alerts for the webhook server
4. Regular database maintenance

### Security:
1. Use HTTPS in production
2. Implement webhook authentication
3. Regular security updates
4. Monitor for unusual activity

### Performance:
1. Limit the number of alerts to prevent overload
2. Use appropriate timeframes for your strategy
3. Regular cleanup of old data
4. Monitor server resources

## Example Alert Configurations

### Basic Price Alert:
- **Condition**: Close crossing above Previous close
- **Frequency**: Every 5 minutes
- **Message**: Standard JSON format

### Technical Indicator Alert:
- **Condition**: RSI(14) crossing above 30
- **Frequency**: Every time
- **Message**: Standard JSON format

### Volume Alert:
- **Condition**: Volume greater than SMA(20, Volume)
- **Frequency**: Every time
- **Message**: Standard JSON format

## Support

If you encounter issues:
1. Check the webhook server logs
2. Verify TradingView alert configuration
3. Test with simple conditions first
4. Ensure network connectivity
5. Review JSON formatting

For persistent issues, create an issue with:
- TradingView alert configuration
- Webhook server logs
- Sample webhook payload
- Error messages