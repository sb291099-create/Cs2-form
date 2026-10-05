import pandas as pd

from cs2form.news import latest_roster, parse_rss, team_news

RSS = """<?xml version="1.0"?><rss><channel>
<item><title>PARIVISION bench slaxejezzz, sign HObbit</title><description>&lt;p&gt;The CIS side made a change&lt;/p&gt;</description>
<link>https://www.hltv.org/news/1/x</link><pubDate>Wed, 17 Sep 2026 10:00:00 +0000</pubDate></item>
<item><title>Vitality win event</title><description>no</description><link>https://www.hltv.org/news/2/y</link>
<pubDate>Thu, 02 Oct 2026 10:00:00 +0000</pubDate></item></channel></rss>"""


def test_rss_and_team_news():
    rss = parse_rss(RSS)
    assert list(rss["date"]) == ["2026-09-17", "2026-10-02"]
    media = pd.DataFrame(
        [
            dict(
                date="2026-10-04",
                title="Jame on Spirit loss",
                link="l1",
                team="parivision",
                player="Jame",
                source="HLTV",
            )
        ]
    )
    items = team_news("parivision", "PARIVISION", rss, media, "2026-09-01")
    assert [i["link"] for i in items] == ["l1", "https://www.hltv.org/news/1/x"]
    assert team_news("furia", "FURIA", rss, media, "2026-09-01") == []


def test_latest_roster_cleans_notes():
    r = pd.DataFrame(
        [
            dict(
                team_name="PARIVISION",
                players="Jame,HObbit",
                notes="{{Notes |{{Notes|September 17th - '''[[PARIVISION]]''' bench {{player|slaxejezzz|flag=ru}}. }} }}",
            )
        ]
    )
    out = latest_roster("PARIVISION", r)
    assert out["players"] == "Jame,HObbit" and "slaxejezzz" in out["notes"] and "{{" not in out["notes"]
