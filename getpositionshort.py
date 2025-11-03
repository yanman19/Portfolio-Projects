import numpy as np
import pandas as pd
from scipy.optimize import brentq

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
    - As price moves against you (upward), you add to position and accumulate losses
    
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
    
    def position_at_price(price: float, max_size: float) -> float:
        """Calculate position size at a given price."""
        if price <= p50:
            return 0.0
        elif price <= p90:
            return ((price - p50) / (p90 - p50)) * max_size
        else:
            return max_size
    
    def calculate_pnl_at_price_analytical(target_price: float, max_size: float) -> float:
        """
        Calculate PnL at target_price starting from current_level analytically.
        """
        if target_price == current_level:
            return 0.0
        
        pnl = 0.0
        
        if target_price > current_level:
            # Moving up (losing money on short)
            # Case 1: Both current and target between p50 and p90
            if p50 < current_level <= p90 and p50 < target_price <= p90:
                # PnL = -∫[current to target] (p - p50) / (p90 - p50) * max_size dp
                pnl = -max_size / (p90 - p50) * (
                    (target_price - p50)**2 / 2 - (current_level - p50)**2 / 2
                )
            # Case 2: Current between p50 and p90, target above p90
            elif p50 < current_level <= p90 and target_price > p90:
                # PnL from current to p90
                pnl_to_p90 = -max_size / (p90 - p50) * (
                    (p90 - p50)**2 / 2 - (current_level - p50)**2 / 2
                )
                # PnL from p90 to target
                pnl_above_p90 = -max_size * (target_price - p90)
                pnl = pnl_to_p90 + pnl_above_p90
            # Case 3: Both current and target above p90
            elif current_level > p90 and target_price > p90:
                pnl = -max_size * (target_price - current_level)
                
        else:  # target_price < current_level
            # Moving down (making money on short)
            # Case 1: Both current and target between p50 and p90
            if p50 < target_price <= p90 and p50 < current_level <= p90:
                # PnL = ∫[target to current] (p - p50) / (p90 - p50) * max_size dp
                pnl = max_size / (p90 - p50) * (
                    (current_level - p50)**2 / 2 - (target_price - p50)**2 / 2
                )
            # Case 2: Target below p50, current between p50 and p90
            elif target_price <= p50 and p50 < current_level <= p90:
                # PnL from p50 to current
                pnl_from_p50 = max_size / (p90 - p50) * (
                    (current_level - p50)**2 / 2
                )
                pnl = pnl_from_p50
            # Case 3: Current above p90, target between p50 and p90
            elif current_level > p90 and p50 < target_price <= p90:
                # PnL from p90 to current
                pnl_above = max_size * (current_level - p90)
                # PnL from target to p90
                pnl_to_p90 = max_size / (p90 - p50) * (
                    (p90 - p50)**2 / 2 - (target_price - p50)**2 / 2
                )
                pnl = pnl_above + pnl_to_p90
        
        return pnl
    
    def calculate_cumulative_pnl_from_current(max_size: float, price_levels: np.ndarray, current_price: float) -> np.ndarray:
        """
        Calculate cumulative PnL starting from current price.
        
        As price moves:
        - UP (against short): Position held loses money, then we add more size
        - DOWN (with short): Position held makes money, then we reduce size
        """
        cum_pnl = np.zeros_like(price_levels, dtype=float)
        
        # Find index of current price
        current_idx = np.argmin(np.abs(price_levels - current_price))
        cum_pnl[current_idx] = 0.0  # PnL at current level is 0
        
        # Calculate PnL moving UP from current (price increases - losses accumulate)
        for i in range(current_idx + 1, len(price_levels)):
            # Position held during this price move
            pos_held = position_at_price(price_levels[i-1], max_size)
            # Price change (positive)
            price_change = price_levels[i] - price_levels[i-1]
            # PnL: negative for shorts when price rises
            pnl_increment = -pos_held * price_change
            cum_pnl[i] = cum_pnl[i-1] + pnl_increment
        
        # Calculate PnL moving DOWN from current (price decreases - profits accumulate)
        for i in range(current_idx - 1, -1, -1):
            # Position held during this price move
            pos_held = position_at_price(price_levels[i+1], max_size)
            # Price change (positive when going down)
            price_change = price_levels[i+1] - price_levels[i]
            # PnL: positive for shorts when price falls
            pnl_increment = pos_held * price_change
            cum_pnl[i] = cum_pnl[i+1] + pnl_increment
        
        return cum_pnl
    
    def calculate_pnl_at_p95_analytical(max_size: float) -> float:
        """
        Calculate PnL at p95 analytically using integration.
        This is more accurate than discretization.
        """
        pnl = 0.0
        
        # Case 1: Current is between p50 and p90
        if p50 < current_level <= p90:
            # PnL from current to p90 (linearly increasing position)
            # position(p) = (p - p50) / (p90 - p50) * max_size
            # PnL = -∫[current to p90] position(p) dp
            # = -max_size / (p90 - p50) * ∫[current to p90] (p - p50) dp
            # = -max_size / (p90 - p50) * [(p - p50)²/2] from current to p90
            pnl_to_p90 = -max_size / (p90 - p50) * (
                (p90 - p50)**2 / 2 - (current_level - p50)**2 / 2
            )
            pnl += pnl_to_p90
            
            # PnL from p90 to p95 (constant max position)
            pnl_to_p95 = -max_size * (p95 - p90)
            pnl += pnl_to_p95
            
        elif current_level > p90:
            # PnL from current to p95 (constant max position)
            pnl = -max_size * (p95 - current_level)
            
        else:  # current_level <= p50
            # Should not happen in typical use case
            # Calculate as if starting from p50
            pnl_to_p90 = -max_size / (p90 - p50) * (p90 - p50)**2 / 2
            pnl_to_p95 = -max_size * (p95 - p90)
            pnl = pnl_to_p90 + pnl_to_p95
        
        return pnl
    
    def objective_function(max_size: float) -> float:
        """Objective: cumulative PnL at p95 should equal -max_loss."""
        pnl_at_p95 = calculate_pnl_at_p95_analytical(max_size)
        return pnl_at_p95 + max_loss
    
    # Solve for max_size using numerical optimization
    # Rough bounds for max_size
    lower_bound = max_loss / (p95 - p50) * 0.01
    upper_bound = max_loss / (p95 - p50) * 100.0
    
    try:
        max_size_solution = brentq(objective_function, lower_bound, upper_bound, xtol=1e-8)
    except ValueError:
        # Fallback with wider bounds
        lower_bound = 1.0
        upper_bound = max_loss * 10
        max_size_solution = brentq(objective_function, lower_bound, upper_bound, xtol=1e-8)
    
    # Calculate initial position at current level
    initial_position = position_at_price(current_level, max_size_solution)
    
    # Generate detailed table
    percentiles = []
    prices = []
    
    # Map percentiles to prices (linear interpolation)
    for ptile in range(50, 96, percentile_step):
        percentiles.append(ptile)
        if ptile <= 50:
            prices.append(p50)
        elif ptile <= 90:
            price = p50 + (ptile - 50) / (90 - 50) * (p90 - p50)
            prices.append(price)
        else:  # 90 < ptile <= 95
            price = p90 + (ptile - 90) / (95 - 90) * (p95 - p90)
            prices.append(price)
    
    # Ensure p95 is included
    if percentiles[-1] != 95:
        percentiles.append(95)
        prices.append(p95)
    
    # Add current level if not already there
    if current_level not in prices:
        # Estimate percentile for current level
        if current_level <= p50:
            current_ptile = 50
        elif current_level <= p90:
            current_ptile = 50 + (current_level - p50) / (p90 - p50) * 40
        else:
            current_ptile = 90 + (current_level - p90) / (p95 - p90) * 5
        
        # Insert in sorted order
        insert_idx = np.searchsorted(prices, current_level)
        percentiles.insert(insert_idx, current_ptile)
        prices.insert(insert_idx, current_level)
    
    prices = np.array(prices)
    
    # Calculate positions at each price
    positions = np.array([position_at_price(p, max_size_solution) for p in prices])
    
    # Calculate cumulative PnL at each price using analytical method
    cum_pnl = np.array([calculate_pnl_at_price_analytical(p, max_size_solution) for p in prices])
    
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
    print("=" * 80)
    print("EXAMPLE: Current level at 50 (between p50 and p90)")
    print("=" * 80)
    
    result = calculate_short_position_with_risk_management(
        current_level=50,
        p50=40,
        p90=90,
        p95=95,
        max_loss=1000000
    )
    
    print(f"\nInitial Position at Current Level (50): {result['initial_position']}")
    print(f"Max Position at P90 (90): {result['max_position']}")
    print(f"\nKey Validation Points:")
    
    table = result['table']
    pnl_at_current = table[table['price'] == 50]['pnl'].values[0]
    pnl_at_p95 = table[table['percentile'] == 95]['pnl'].values[0]
    pnl_at_p50 = table[table['percentile'] == 50]['pnl'].values[0]
    
    print(f"  - PnL at current level (50): ${pnl_at_current:,.2f} (should be $0)")
    print(f"  - PnL at p95 (95): ${pnl_at_p95:,.2f} (should be ~$-{1000000:,})")
    print(f"  - PnL at p50 (40): ${pnl_at_p50:,.2f} (should be positive)")
    
    print("\n\nDetailed Table:")
    print(result['table'].to_string(index=False))
