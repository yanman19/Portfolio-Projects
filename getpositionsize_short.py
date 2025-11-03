import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq

def calculate_short_position_with_risk_management(
    current_level: float,
    p50: float,
    p90: float,
    p95: float,
    max_loss: float,
    percentile_step: int = 5
):
    """
    Calculate optimal short position sizing with risk management constraints.
    
    Strategy:
    - Flatten entire position at p50 (position = 0)
    - Linear position scaling between p50 and p90
    - Max position reached at p90
    - Stop out at p95 (max loss reached)
    
    Args:
        current_level: Current market price
        p50: Price at 50th percentile (target - flatten position here)
        p90: Price at 90th percentile (max position size reached)
        p95: Price at 95th percentile (stop loss level)
        max_loss: Maximum acceptable cumulative loss
        percentile_step: Step size for percentile increments in output table
    
    Returns:
        dict with:
            - initial_position: Position size to take at current level
            - max_position: Maximum position size at p90
            - table: DataFrame with percentile, price, position, pnl
    """
    
    def calculate_cumulative_pnl(max_size: float, price_levels: np.ndarray, current_idx: int):
        """Calculate cumulative PnL for given max position size across price levels."""
        positions = np.zeros_like(price_levels, dtype=float)
        cum_pnl = np.zeros_like(price_levels, dtype=float)
        max_size = float(max_size)  # Ensure scalar
        
        # First, calculate all positions
        for i, price in enumerate(price_levels):
            price_val = float(price)
            # Determine position based on price level
            if price_val <= p50:
                positions[i] = 0.0  # Flattened below p50
            elif price_val <= p90:
                # Linear interpolation between p50 and p90
                positions[i] = ((price_val - p50) / (p90 - p50)) * max_size
            else:  # price > p90
                positions[i] = max_size  # Max position above p90
        
        # Set PnL at current level
        cum_pnl[current_idx] = 0
        
        # Calculate PnL moving up from current (towards p95)
        for i in range(current_idx + 1, len(price_levels)):
            price_change = price_levels[i] - price_levels[i-1]  # Positive
            pnl_increment = -positions[i-1] * price_change  # Negative for shorts when price rises
            cum_pnl[i] = cum_pnl[i-1] + pnl_increment
        
        # Calculate PnL moving down from current (towards p50) - iterate backwards
        for i in range(current_idx - 1, -1, -1):
            price_change = price_levels[i+1] - price_levels[i]  # Positive
            pnl_increment = positions[i+1] * price_change  # Positive for shorts when price falls
            cum_pnl[i] = cum_pnl[i+1] + pnl_increment
        
        return positions, cum_pnl
    
    def objective_function(max_size: float):
        """Objective: cumulative PnL at p95 should equal -max_loss."""
        # Create fine price grid from p50 to p95, ensuring current level is included
        temp_prices = np.sort(np.unique(np.concatenate([
            np.linspace(p50, current_level, 100),
            np.linspace(current_level, p95, 100)
        ])))
        current_idx = np.where(np.abs(temp_prices - current_level) < 1e-10)[0][0]
        
        _, cum_pnl = calculate_cumulative_pnl(max_size, temp_prices, current_idx)
        return cum_pnl[-1] + max_loss  # Should be zero when PnL = -max_loss
    
    # Solve for max_size that gives us exactly -max_loss at p95
    # Use brentq for more robust root finding
    # Bounds: reasonable range for max_size
    lower_bound = max_loss / (p95 - p50) * 0.1  # Conservative lower bound
    upper_bound = max_loss / (p95 - p50) * 10.0  # Conservative upper bound
    
    try:
        max_size_solution = brentq(objective_function, lower_bound, upper_bound, xtol=1e-8)
    except ValueError:
        # If brentq fails (e.g., root not in bounds), fall back to fsolve
        initial_guess = max_loss / (p95 - current_level)
        max_size_solution = fsolve(objective_function, initial_guess, full_output=False, xtol=1e-10)[0]
    
    # Calculate initial position at current level
    initial_position = ((current_level - p50) / (p90 - p50)) * max_size_solution
    
    # Generate detailed table with specified percentile steps
    # Create percentile range from p50 to p95
    percentiles = []
    prices = []
    
    # Determine percentile mapping (assuming p50=50th, p90=90th, p95=95th)
    # Linear interpolation for intermediate percentiles
    for ptile in range(50, 96, percentile_step):
        percentiles.append(ptile)
        if ptile <= 50:
            prices.append(p50)
        elif ptile <= 90:
            # Linear interpolation between p50 and p90
            price = p50 + (ptile - 50) / (90 - 50) * (p90 - p50)
            prices.append(price)
        elif ptile <= 95:
            # Linear interpolation between p90 and p95
            price = p90 + (ptile - 90) / (95 - 90) * (p95 - p90)
            prices.append(price)
    
    # Add p95 if not already included
    if percentiles[-1] != 95:
        percentiles.append(95)
        prices.append(p95)
    
    # Add current level if it's not in the list
    if current_level not in prices:
        # Insert current level in sorted order
        insert_idx = np.searchsorted(prices, current_level)
        # Estimate percentile for current level
        if current_level <= p50:
            current_ptile = 50
        elif current_level <= p90:
            current_ptile = 50 + (current_level - p50) / (p90 - p50) * 40
        else:
            current_ptile = 90 + (current_level - p90) / (p95 - p90) * 5
        
        percentiles.insert(insert_idx, current_ptile)
        prices.insert(insert_idx, current_level)
    
    prices = np.array(sorted(prices))
    
    # Find index of current level in prices array
    current_idx_display = np.argmin(np.abs(prices - current_level))
    
    # Calculate positions and PnL directly on display grid
    positions, cum_pnl = calculate_cumulative_pnl(max_size_solution, prices, current_idx_display)
    
    # Create output table
    table = pd.DataFrame({
        'percentile': percentiles,
        'price': prices,
        'position': positions,
        'pnl': cum_pnl
    })
    
    # Round for display
    table['percentile'] = table['percentile'].round(1)
    table['price'] = table['price'].round(2)
    table['position'] = table['position'].round(2)
    table['pnl'] = table['pnl'].round(2)
    
    return {
        'initial_position': round(initial_position, 2),
        'max_position': round(max_size_solution, 2),
        'table': table
    }


# Example usage
if __name__ == "__main__":
    result = calculate_short_position_with_risk_management(
        current_level=100,
        p50=95,
        p90=105,
        p95=108,
        max_loss=5000
    )
    
    print(f"Initial Position at Current Level: {result['initial_position']}")
    print(f"Max Position at P90: {result['max_position']}")
    print("\nDetailed Table:")
    print(result['table'].to_string(index=False))
