# 🤖 Polymarket ML Integration Guide

This guide covers the machine learning capabilities integrated into the Polymarket LangGraph agent system.

## 🎯 **Overview**

The system now includes **supervised learning for probability calibration and edge detection** using Gradient Boosting models (XGBoost or scikit-learn fallback). This provides calibrated probability estimates to identify betting opportunities with statistical edges.

## 🏗️ **Architecture**

### **ML Strategy Framework**
- **Base Class**: `MLBettingStrategy` in `agents/ml_strategies/base_strategy.py`
- **Implementation**: `XGBoostProbabilityStrategy` in `agents/ml_strategies/xgboost_strategy.py`
- **Integration**: Planning agent in `agents/graph/planning_agent.py`

### **Data Pipeline**
- **Ingestion**: `PolymarketDataIngestion` in `agents/automl/data_ingestion.py`
- **Training**: Automated pipeline with feature engineering
- **Evaluation**: Backtesting and performance metrics

## 🚀 **Quick Start**

### **1. Train a Model**
```bash
# Train on real market data (requires resolved markets in database)
python scripts/python/ml_pipeline_cli.py train-xgboost --days-back 365 --min-volume 1000

# Or test with synthetic data
python train_xgboost_strategy.py --test
```

### **2. Use in Planning Agent**
```bash
# Enable sports focus + ML predictions
export MARKET_FOCUS=sports
python scripts/python/cli.py run-memory-agent "Find high-volume NFL markets with ML edge"
```

### **3. Check Performance**
The trained model provides:
- **Calibrated probabilities** (not just market prices)
- **Edge calculations** (expected value per dollar)
- **Confidence scores** and position sizing recommendations

## 📊 **Features Used**

The model learns from these market characteristics:

| Feature | Description | Importance |
|---------|-------------|------------|
| **Price Features** | Current YES/NO prices, distance from fair odds | High |
| **Volume Features** | Raw volume, log volume, volume categories | High |
| **Time Features** | Days to resolution, market age | Medium |
| **Liquidity** | Trading liquidity metrics | High |
| **Category** | Sports, politics, crypto, etc. (one-hot encoded) | Medium |
| **Text Features** | Question length, keyword presence | Low-Medium |

## 🎯 **How It Works**

### **Training Phase**
1. **Data Collection**: Fetch resolved markets from Polymarket API
2. **Feature Engineering**: Transform raw data into ML features
3. **Model Training**: Gradient boosting on historical outcomes
4. **Validation**: Cross-validation and backtesting

### **Prediction Phase**
1. **Feature Extraction**: Same engineering as training
2. **Probability Estimation**: Model predicts true P(YES)
3. **Edge Calculation**: Compare to market price
4. **Betting Decision**: Kelly criterion position sizing

### **Integration with Agents**
- **Planning Agent**: Uses ML predictions alongside LLM estimates
- **Memory Agent**: Focuses search on high-edge markets
- **Trading Tools**: Position sizing based on ML confidence

## 📈 **Performance Metrics**

Track these to evaluate model quality:

### **Calibration Metrics**
- **ROC-AUC**: >0.7 indicates good discriminative ability
- **Brier Score**: <0.25 indicates good calibration
- **Log Loss**: Lower is better

### **Trading Metrics**
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Positive Edge %**: Fraction of markets with >0% expected value

## 🛠️ **Usage Examples**

### **Basic Training**
```python
from agents.ml_strategies.xgboost_strategy import XGBoostProbabilityStrategy

# Initialize strategy
strategy = XGBoostProbabilityStrategy()

# Run full pipeline
results = strategy.run_full_pipeline(
    days_back=365,
    min_volume=1000,
    test_size=0.2
)

print(f"Model ROC-AUC: {results['evaluation_metrics']['roc_auc']:.3f}")
print(f"Backtest Return: {results['backtest_results']['total_return_pct']:.1f}%")
```

### **Single Market Prediction**
```python
# Load trained model
strategy.load_model("data/models/gradient_boosting_model.pkl")

# Predict on a market
market_data = {
    'id': 'market_123',
    'question': 'Will Team A win?',
    'outcome_prices': ['0.55', '0.45'],
    'volume': 50000,
    'liquidity': 10000,
    'category': 'sports'
}

result = strategy.predict(market_data)
print(f"Predicted prob: {result.predicted_probability:.3f}")
print(f"Recommended bet: {result.recommended_bet}")
print(f"Position size: {result.position_size:.3f}")
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Focus on specific categories
export MARKET_FOCUS=sports  # Only sports markets
export MARKET_FOCUS=politics  # Only politics markets
# unset or empty = all categories

# Model paths
export GRADIENT_BOOSTING_MODEL_PATH="data/models/custom_model.pkl"
```

### **Model Hyperparameters**
```python
hyperparams = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

strategy.train(training_data, hyperparams=hyperparams)
```

## 📚 **Advanced Topics**

### **Feature Engineering**
The system automatically engineers features, but you can extend:

```python
class CustomStrategy(XGBoostProbabilityStrategy):
    def prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        # Add custom features
        features = super().prepare_features(market_data)

        # Add sentiment score
        sentiment = analyze_market_sentiment(market_data['question'])
        features = np.append(features, sentiment)

        # Add momentum features
        momentum = calculate_price_momentum(market_data['id'])
        features = np.append(features, momentum)

        return features
```

### **Ensemble Methods**
Combine multiple models for better performance:

```python
class EnsembleStrategy(MLBettingStrategy):
    def __init__(self):
        self.models = [
            XGBoostProbabilityStrategy(),
            LSTMT imeSeriesStrategy(),
            SentimentStrategy()
        ]

    def predict(self, market_data):
        # Average predictions
        predictions = [model.predict(market_data).predicted_probability
                      for model in self.models]
        ensemble_prob = np.mean(predictions)

        # Calculate confidence as 1 - variance
        confidence = 1 - np.var(predictions)
        # ... rest of ensemble logic
```

### **Backtesting Framework**
```python
# Evaluate on historical data
backtest_results = strategy.backtest_strategy(
    historical_markets=historical_data,
    initial_capital=10000.0
)

print(f"Total return: {backtest_results['total_return_pct']:.1f}%")
print(f"Win rate: {backtest_results['win_rate']:.1%}")
print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
```

## 🚨 **Important Notes**

### **Data Requirements**
- **Minimum 100 resolved markets** for meaningful training
- **Time-based splits** to avoid lookahead bias
- **Resolved outcomes only** for supervised learning

### **Risk Management**
- **Kelly Criterion**: Conservative position sizing (½ Kelly)
- **Maximum drawdown limits**: Stop trading if losses exceed threshold
- **Regular retraining**: Markets change, models decay

### **Limitations**
- **Past performance ≠ future results**
- **Market efficiency**: Polymarket is generally efficient
- **Transaction costs**: Not fully modeled yet
- **Liquidity constraints**: Thin markets have wide spreads

### **Ethical Considerations**
- **No guaranteed profits**: ML enhances but doesn't guarantee success
- **Responsible gambling**: Only bet what you can afford to lose
- **Transparency**: Always understand what the model is doing

## 🔮 **Future Enhancements**

### **Short Term**
- [ ] **Time series features**: Price momentum, volume trends
- [ ] **News sentiment integration**: Real-time news analysis
- [ ] **Cross-market correlations**: Related market influences
- [ ] **Hyperparameter optimization**: Automated tuning

### **Medium Term**
- [ ] **Reinforcement learning**: Policy-based trading agents
- [ ] **Multi-outcome modeling**: Categorical predictions
- [ ] **Live performance monitoring**: Real-time model updates
- [ ] **Portfolio optimization**: Multi-market position management

### **Long Term**
- [ ] **Market microstructure**: Order book analysis
- [ ] **Alternative data**: Social media, satellite imagery
- [ ] **Generative models**: Synthetic data augmentation
- [ ] **Causal inference**: Understanding market drivers

## 📞 **Support**

- **Check model performance** regularly with backtesting
- **Retrain monthly** as new resolved markets become available
- **Monitor feature importance** to understand what the model learns
- **Start conservative** with small position sizes

---

**Remember**: This is a sophisticated tool, but successful trading requires discipline, risk management, and continuous learning. The ML models enhance decision-making but don't replace human judgment.