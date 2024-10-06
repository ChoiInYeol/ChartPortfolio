from Data.generate_chart import GenerateStockData

if __name__ == "__main__":
    # 데이터를 생성할 연도 범위 설정
    year_list = list(range(2001, 2024))
    
    # 차트 타입 설정 (bar, pixel, centered_pixel 중 선택)
    chart_type = "bar"
    
    # 윈도우 크기 설정 (60일)
    ws = 60
    
    # 데이터 빈도 설정 (week, month, quarter 중 선택)
    freq = "month"
    
    # 이동평균선 설정 (여기서는 윈도우 크기와 동일하게 설정)
    ma_lags = [ws]
    
    # 거래량 바 포함 여부
    vb = True
    
    # 각 연도별로 데이터 생성
    for year in year_list:
        print(f"{ws}D {freq} {chart_type} {year}")
        
        # GenerateStockData 객체 생성
        chart_generator = GenerateStockData(
            "USA",  # 국가 설정
            year,   # 연도
            ws,     # 윈도우 크기
            freq,   # 데이터 빈도
            chart_freq=1,  # 차트 빈도 (1은 모든 데이터 포인트 사용)
            ma_lags=ma_lags,  # 이동평균선 설정
            volume_bar=vb,    # 거래량 바 포함 여부
            chart_type=chart_type,  # 차트 타입
        )
        
        # CNN2D 데이터 생성 및 저장
        chart_generator.save_annual_data()

