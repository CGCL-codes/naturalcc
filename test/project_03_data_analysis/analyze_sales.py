"""
项目: project_03_data_analysis
测试主题: Python 数据处理与可视化
功能: 销售数据分析脚本，使用 pandas 处理数据并生成可视化报告
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """加载 CSV 数据"""
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"已加载 {len(df)} 条记录")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清理数据：处理缺失值和异常"""
    # 删除空值
    df = df.dropna(subset=['amount', 'product'])
    # 过滤异常金额（负数和过大值）
    df = df[(df['amount'] > 0) & (df['amount'] < 10000)]
    return df


def analyze_by_product(df: pd.DataFrame) -> pd.DataFrame:
    """按商品汇总销售"""
    summary = df.groupby('product').agg(
        total_amount=('amount', 'sum'),
        avg_amount=('amount', 'mean'),
        count=('amount', 'count')
    ).sort_values('total_amount', ascending=False)
    return summary


def analyze_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """按月汇总销售"""
    df = df.copy()
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['amount'].sum()
    return monthly


def plot_results(by_product: pd.DataFrame, by_month: pd.Series, output_dir: str):
    """生成可视化图表"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 商品销售柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    by_product['total_amount'].head(10).plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Top 10 商品销售总额')
    ax.set_xlabel('商品')
    ax.set_ylabel('销售额')
    plt.tight_layout()
    plt.savefig(output_path / 'top_products.png', dpi=100)
    plt.close()

    # 月度销售趋势图
    fig, ax = plt.subplots(figsize=(10, 6))
    by_month.plot(kind='line', ax=ax, marker='o', color='coral')
    ax.set_title('月度销售趋势')
    ax.set_xlabel('月份')
    ax.set_ylabel('销售额')
    plt.tight_layout()
    plt.savefig(output_path / 'monthly_trend.png', dpi=100)
    plt.close()


def main():
    """主函数"""
    data_file = Path(__file__).parent / 'data' / 'sample_sales.csv'
    output_dir = Path(__file__).parent / 'output'

    df = load_data(data_file)
    df = clean_data(df)

    by_product = analyze_by_product(df)
    by_month = analyze_by_month(df)

    print("\n=== 商品销售汇总 ===")
    print(by_product)
    print("\n=== 月度销售汇总 ===")
    print(by_month)

    plot_results(by_product, by_month, output_dir)
    print(f"\n图表已保存至 {output_dir}")


if __name__ == '__main__':
    main()
