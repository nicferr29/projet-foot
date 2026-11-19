from SoccerNet.Downloader import SoccerNetDownloader

dl = SoccerNetDownloader(LocalDirectory="SoccerNet")
dl.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])
dl.downloadGames(files=["Labels-caption.json"], split=["train", "valid", "test"])