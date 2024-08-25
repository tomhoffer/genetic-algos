import datetime
from asyncio import get_event_loop

from src.tradingbot.exceptions import NoDataFoundException
from src.tradingbot.preprocessor.config import Config
from src.tradingbot.preprocessor.loader import Ticker
from src.tradingbot.repository import TradingdataRepository


async def main():
    trading_data_repository = TradingdataRepository()
    await trading_data_repository.init_pool(
        dbname=Config.get_value("POSTGRES_DB"),
        user=Config.get_value("POSTGRES_USER"),
        password=Config.get_value("POSTGRES_PASSWORD"),
        host=Config.get_value("POSTGRES_HOST"),
    )

    # Find the timestamp of the latest data record in the DB
    end_timestamp_str: str = datetime.date.today().strftime("%Y-%m-%d")
    try:
        latest_record_timestamp: datetime.datetime = (await trading_data_repository.get_latest_record())[0]
    except NoDataFoundException:
        latest_record_timestamp: datetime.date = datetime.date.fromisoformat('2019-01-01')
        print(f"No data found in the DB, fetching history since {latest_record_timestamp}")

    # Prefetch previous days, so it is possible to compute technical indicators
    number_of_days_to_prefetch = 30
    start_timestamp_str: str = (latest_record_timestamp - datetime.timedelta(
        days=number_of_days_to_prefetch + 2)).strftime(
        "%Y-%m-%d")

    # Fill in the missing data
    ticker = Ticker(Config.get_value("TICKER_NAME"))
    ticker.fetch_historical_data(start=start_timestamp_str, end=end_timestamp_str)
    ticker.compute_technical_indicators()

    # Remove the prefetched data before uploading
    print("Before transformation: ", ticker.data_df)
    ticker.data_df.drop(ticker.data_df.head(number_of_days_to_prefetch + 3).index, inplace=True)
    print("After transformation: ", ticker.data_df)

    # Fetch sentiment data only for relevant dates
    ticker.fetch_sentiment_data()

    # Upload the data into DB
    trading_data_repository.upload_from_df(df=ticker.data_df, if_exists='append')
    print(f"{len(ticker.data_df)} records have been uploaded.")

    await trading_data_repository.close_pool()


loop = get_event_loop()
loop.run_until_complete(main())
