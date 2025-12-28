# Polymarket API Reference

Complete reference for all Polymarket APIs - REST endpoints, WebSocket channels, and Python client methods.

## Table of Contents

1. [API Endpoints Overview](#api-endpoints-overview)
2. [Gamma API - Market Discovery](#gamma-api---market-discovery)
3. [Sports API - Teams & Leagues](#sports-api---teams--leagues)
4. [CLOB API - Trading](#clob-api---trading)
5. [Data API - User Data](#data-api---user-data)
6. [WebSocket API - Real-time Data](#websocket-api---real-time-data)
7. [Python Client Methods](#python-client-methods)

---

## API Endpoints Overview

| API | Base URL | Purpose |
|-----|----------|---------|
| **Gamma API** | `https://gamma-api.polymarket.com` | Market discovery, events, metadata |
| **CLOB API** | `https://clob.polymarket.com` | Trading, order management, order books |
| **Data API** | `https://data-api.polymarket.com` | User holdings, on-chain activity |
| **WebSocket** | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Real-time order book, trades |
| **RTDS** | `wss://ws-live-data.polymarket.com` | Crypto prices, comments |

---

## Gamma API - Market Discovery

Base URL: `https://gamma-api.polymarket.com`

### Three Strategies for Fetching Markets

There are three recommended approaches, each optimized for different use cases:

| Strategy | Use Case | Endpoint |
|----------|----------|----------|
| **By Slug** | Fetch specific individual markets/events | `GET /events/slug/{slug}` |
| **By Tags** | Filter by category or sport | `GET /markets?tag_id=X` |
| **Via Events** | Retrieve all active markets | `GET /events?closed=false` |

---

### Strategy 1: Fetch by Slug (Best for Specific Markets)

Extract the slug from any Polymarket URL:
```
https://polymarket.com/event/fed-decision-in-october?tid=1758818660485
                            ↑
                  Slug: fed-decision-in-october
```

**For Events:**
```bash
curl "https://gamma-api.polymarket.com/events/slug/fed-decision-in-october"
```

**For Markets:**
```bash
curl "https://gamma-api.polymarket.com/markets/slug/will-trump-win-2024"
```

---

### Strategy 2: Fetch by Tags (Best for Categories)

**Discover Available Tags:**
```bash
# General tags
curl "https://gamma-api.polymarket.com/tags"

# Sports tags with metadata
curl "https://gamma-api.polymarket.com/sports"
```

**Filter by Tag:**
```bash
curl "https://gamma-api.polymarket.com/events?tag_id=100381&limit=10&closed=false"
```

**Additional Tag Options:**
- `related_tags=true` - Include related tag markets
- `exclude_tag_id=X` - Exclude specific tags

---

### Strategy 3: Fetch All Active Markets (Best for Discovery)

Use the events endpoint (most efficient - events contain their markets):

```bash
curl "https://gamma-api.polymarket.com/events?order=id&ascending=false&closed=false&limit=100"
```

**Key Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| `order` | `id` | Order by event ID |
| `ascending` | `false` | Newest events first |
| `closed` | `false` | Only active markets |
| `limit` | `50` | Results per page |
| `offset` | `0` | Pagination offset |

---

### Pagination

For large datasets, paginate with `limit` and `offset`:

```bash
# Page 1: First 50 results
curl "https://gamma-api.polymarket.com/events?order=id&ascending=false&closed=false&limit=50&offset=0"

# Page 2: Next 50 results
curl "https://gamma-api.polymarket.com/events?order=id&ascending=false&closed=false&limit=50&offset=50"

# Page 3: Next 50 results
curl "https://gamma-api.polymarket.com/events?order=id&ascending=false&closed=false&limit=50&offset=100"
```

**With tag filtering:**
```bash
curl "https://gamma-api.polymarket.com/markets?tag_id=100381&closed=false&limit=25&offset=0"
curl "https://gamma-api.polymarket.com/markets?tag_id=100381&closed=false&limit=25&offset=25"
```

---

### List Markets

```bash
GET /markets
```

**Query Parameters - Pagination & Sorting:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Results per page (>= 0) |
| `offset` | integer | Pagination offset (>= 0) |
| `order` | string | Comma-separated fields to order by |
| `ascending` | boolean | Sort direction |

**Query Parameters - Filtering by ID:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | integer[] | Filter by market IDs |
| `slug` | string[] | Filter by market slugs |
| `clob_token_ids` | string[] | Filter by CLOB token IDs |
| `condition_ids` | string[] | Filter by condition IDs |
| `question_ids` | string[] | Filter by question IDs |
| `market_maker_address` | string[] | Filter by market maker address |

**Query Parameters - Tag & Category:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tag_id` | integer | Filter by tag ID |
| `related_tags` | boolean | Include related tag markets |
| `include_tag` | boolean | Include tag data in response |

**Query Parameters - Volume & Liquidity:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `liquidity_num_min` | number | Minimum liquidity |
| `liquidity_num_max` | number | Maximum liquidity |
| `volume_num_min` | number | Minimum volume |
| `volume_num_max` | number | Maximum volume |

**Query Parameters - Date Ranges:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date_min` | datetime | Start date minimum |
| `start_date_max` | datetime | Start date maximum |
| `end_date_min` | datetime | End date minimum |
| `end_date_max` | datetime | End date maximum |

**Query Parameters - Sports:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sports_market_types` | string[] | Filter by sports market types (moneyline, spreads, totals, etc.) |
| `game_id` | string | Filter by game ID |

**Query Parameters - Other:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `closed` | boolean | Filter by closed status |
| `cyom` | boolean | Create Your Own Market |
| `uma_resolution_status` | string | UMA resolution status |
| `rewards_min_size` | number | Minimum rewards size |

**Response Fields (Key):**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique market identifier |
| `question` | string | Market question |
| `conditionId` | string | Condition ID for trading |
| `slug` | string | URL-friendly identifier |
| `description` | string | Market description |
| `outcomes` | string | JSON array of outcome names |
| `outcomePrices` | string | JSON array of current prices |
| `volume` | string | Total volume |
| `volumeNum` | number | Volume as number |
| `liquidity` | string | Total liquidity |
| `liquidityNum` | number | Liquidity as number |
| `active` | boolean | Is market active |
| `closed` | boolean | Is market closed |
| `endDate` | datetime | Market end date |
| `clobTokenIds` | string | CLOB token IDs (comma-separated) |

**Volume Breakdown Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `volume24hr` | number | 24-hour volume |
| `volume1wk` | number | 1-week volume |
| `volume1mo` | number | 1-month volume |
| `volume1yr` | number | 1-year volume |
| `volumeAmm` | number | AMM volume |
| `volumeClob` | number | CLOB volume |
| `volume24hrAmm` / `volume24hrClob` | number | 24hr by venue |

**Liquidity Breakdown:**

| Field | Type | Description |
|-------|------|-------------|
| `liquidityAmm` | number | AMM liquidity |
| `liquidityClob` | number | CLOB liquidity |

**Order Book Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `enableOrderBook` | boolean | Order book enabled |
| `orderPriceMinTickSize` | number | Minimum tick size |
| `orderMinSize` | number | Minimum order size |
| `acceptingOrders` | boolean | Accepting orders |
| `bestBid` | number | Best bid price |
| `bestAsk` | number | Best ask price |
| `spread` | number | Bid-ask spread |
| `lastTradePrice` | number | Last trade price |

**Price Change Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `oneHourPriceChange` | number | 1-hour price change |
| `oneDayPriceChange` | number | 1-day price change |
| `oneWeekPriceChange` | number | 1-week price change |
| `oneMonthPriceChange` | number | 1-month price change |

**Sports-Specific Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `sportsMarketType` | string | Type (moneyline, spreads, totals) |
| `line` | number | Betting line |
| `gameId` | string | Associated game ID |
| `teamAID` / `teamBID` | string | Team identifiers |
| `gameStartTime` | string | Game start time |
| `eventStartTime` | datetime | Event start time |

**Example - Get all active markets:**
```bash
curl "https://gamma-api.polymarket.com/markets?closed=false&limit=100"
```

**Example - Get high-volume markets:**
```bash
curl "https://gamma-api.polymarket.com/markets?volume_num_min=100000&closed=false&limit=50"
```

**Example - Get markets by tag:**
```bash
curl "https://gamma-api.polymarket.com/markets?tag_id=126&closed=false&limit=20"
```

**Example - Get sports markets by type:**
```bash
curl "https://gamma-api.polymarket.com/markets?sports_market_types=moneyline&closed=false&limit=20"
```

**Example - Get market by CLOB token:**
```bash
curl "https://gamma-api.polymarket.com/markets?clob_token_ids=101669189743438912873361127612589311253202068943959811456820079057046819967115"
```

---

### Additional Market Endpoints

#### Get Market by ID

```bash
GET /markets/{market_id}
```

```bash
curl "https://gamma-api.polymarket.com/markets/253123"
```

#### Get Market by Slug

```bash
GET /markets/slug/{slug}
```

```bash
curl "https://gamma-api.polymarket.com/markets/slug/will-trump-win-2024"
```

### Fetching Events

Events are collections of related markets. The events endpoint is the **recommended approach** for discovering markets.

#### List Events

```bash
GET /events
```

**Query Parameters - Pagination & Sorting:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Results per page (>= 0) |
| `offset` | integer | Pagination offset (>= 0) |
| `order` | string | Comma-separated fields to order by |
| `ascending` | boolean | Sort direction |

**Query Parameters - Filtering:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | integer[] | Filter by specific event IDs |
| `slug` | string[] | Filter by event slugs |
| `tag_id` | integer | Filter by tag ID |
| `tag_slug` | string | Filter by tag slug |
| `exclude_tag_id` | integer[] | Exclude specific tags |
| `related_tags` | boolean | Include related tag events |
| `active` | boolean | Filter by active status |
| `archived` | boolean | Filter by archived status |
| `featured` | boolean | Filter featured events |
| `closed` | boolean | Filter by closed status |
| `cyom` | boolean | Create Your Own Market events |
| `recurrence` | string | Filter by recurrence pattern |

**Query Parameters - Volume & Liquidity:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `liquidity_min` | number | Minimum liquidity |
| `liquidity_max` | number | Maximum liquidity |
| `volume_min` | number | Minimum volume |
| `volume_max` | number | Maximum volume |

**Query Parameters - Date Ranges:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_date_min` | datetime | Start date minimum |
| `start_date_max` | datetime | Start date maximum |
| `end_date_min` | datetime | End date minimum |
| `end_date_max` | datetime | End date maximum |

**Query Parameters - Include Options:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_chat` | boolean | Include chat data |
| `include_template` | boolean | Include template data |

**Response Fields (Key):**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique event identifier |
| `ticker` | string | Event ticker symbol |
| `slug` | string | URL-friendly identifier |
| `title` | string | Event title |
| `description` | string | Event description |
| `resolutionSource` | string | Resolution source URL |
| `startDate` | datetime | Event start date |
| `endDate` | datetime | Event end date |
| `image` | string | Event image URL |
| `active` | boolean | Is event active |
| `closed` | boolean | Is event closed |
| `archived` | boolean | Is event archived |
| `featured` | boolean | Is event featured |
| `liquidity` | number | Total liquidity |
| `volume` | number | Total volume |
| `volume24hr` | number | 24-hour volume |
| `volume1wk` | number | 1-week volume |
| `volume1mo` | number | 1-month volume |
| `openInterest` | number | Open interest |
| `commentCount` | integer | Number of comments |
| `markets` | object[] | Array of markets in this event |
| `tags` | object[] | Array of associated tags |
| `series` | object[] | Series information |
| `categories` | object[] | Category assignments |
| `negRisk` | boolean | Negative risk market |
| `enableOrderBook` | boolean | Order book enabled |
| `liquidityAmm` | number | AMM liquidity |
| `liquidityClob` | number | CLOB liquidity |

**Sports-Specific Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `live` | boolean | Is event live |
| `ended` | boolean | Has event ended |
| `score` | string | Current score |
| `elapsed` | string | Time elapsed |
| `period` | string | Current period |
| `gameStatus` | string | Game status |
| `eventDate` | string | Event date |
| `eventWeek` | integer | Event week number |
| `seriesSlug` | string | Series slug |
| `spreadsMainLine` | number | Spreads main line |
| `totalsMainLine` | number | Totals main line |

**Example - Get newest active events:**
```bash
curl "https://gamma-api.polymarket.com/events?order=id&ascending=false&closed=false&limit=100"
```

**Example - Get high-volume events:**
```bash
curl "https://gamma-api.polymarket.com/events?volume_min=100000&closed=false&limit=50"
```

**Example - Get events by tag:**
```bash
curl "https://gamma-api.polymarket.com/events?tag_id=126&closed=false&limit=20"
```

**Example - Get featured events:**
```bash
curl "https://gamma-api.polymarket.com/events?featured=true&closed=false"
```

---

#### Get Event by ID

```bash
GET /events/{event_id}
```

```bash
curl "https://gamma-api.polymarket.com/events/12345"
```

---

#### Get Event by Slug

```bash
GET /events/slug/{slug}
```

**Example:**
```bash
curl "https://gamma-api.polymarket.com/events/slug/fed-decision-in-october"
```

### Tags & Categories

#### List Tags

```bash
GET /tags
```

Returns categorization tags used for filtering markets and events. Tags are the primary mechanism for categorizing content across the platform.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | integer | Results per page (must be >= 0) |
| `offset` | integer | Pagination offset (must be >= 0) |
| `order` | string | Comma-separated list of fields to order by |
| `ascending` | boolean | Sort direction |
| `include_template` | boolean | Include template tags |
| `is_carousel` | boolean | Filter carousel tags only |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique tag identifier (use with `tag_id` parameter) |
| `label` | string \| null | Human-readable tag name |
| `slug` | string \| null | URL-friendly identifier |
| `forceShow` | boolean \| null | Always show this tag |
| `forceHide` | boolean \| null | Always hide this tag |
| `isCarousel` | boolean \| null | Featured in carousel |
| `publishedAt` | string \| null | When tag was published |
| `createdAt` | datetime \| null | Creation timestamp |
| `updatedAt` | datetime \| null | Last update timestamp |

**Example Request:**
```bash
curl "https://gamma-api.polymarket.com/tags?limit=100"
```

**Example Response:**
```json
{
  "id": "126",
  "label": "Trump",
  "slug": "trump",
  "publishedAt": "2023-12-20 18:56:23.087+00",
  "createdAt": "2023-12-20T18:56:23.109Z",
  "updatedAt": "2025-12-18T02:37:54.922815Z",
  "requiresTranslation": false
}
```

**Key Tags for Filtering:**

| Tag ID | Label | Use Case |
|--------|-------|----------|
| `126` | Trump | Trump-related markets |
| `24` | USA Election | US election markets |
| `1101` | US Election | US election (alternate) |
| `235` | Bitcoin | Bitcoin/crypto markets |
| `131` | Interest Rates | Fed/finance markets |
| `414` | Health | Health-related markets |
| `966` | Warpcast | Social/tech markets |
| `1346` | International Conflicts | Geopolitics |
| `439` | AI | Artificial intelligence (carousel) |

**Usage - Filter Markets by Tag:**
```bash
# Get all Trump-related events
curl "https://gamma-api.polymarket.com/events?tag_id=126&closed=false&limit=20"

# Get Bitcoin markets
curl "https://gamma-api.polymarket.com/markets?tag_id=235&closed=false"

# Get carousel/featured tags
curl "https://gamma-api.polymarket.com/tags?is_carousel=true"
```

---

#### Get Tag by ID

```bash
GET /tags/{id}
```

Retrieves a specific tag by its unique identifier.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | integer | Yes | Tag ID |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_template` | boolean | Include template data |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique tag identifier |
| `label` | string \| null | Human-readable tag name |
| `slug` | string \| null | URL-friendly identifier |
| `forceShow` | boolean \| null | Always display this tag |
| `forceHide` | boolean \| null | Always hide this tag |
| `isCarousel` | boolean \| null | Featured in carousel |
| `publishedAt` | string \| null | Publication timestamp |
| `createdBy` | integer \| null | Creator user ID |
| `updatedBy` | integer \| null | Last updater user ID |
| `createdAt` | datetime \| null | Creation timestamp |
| `updatedAt` | datetime \| null | Last update timestamp |

**Example Request:**
```bash
curl "https://gamma-api.polymarket.com/tags/126"
```

**Example Response:**
```json
{
  "id": "126",
  "label": "Trump",
  "slug": "trump",
  "forceShow": false,
  "publishedAt": "2023-11-02 21:23:16.384+00",
  "updatedBy": 15,
  "createdAt": "2023-11-02T21:23:16.39Z",
  "updatedAt": "2025-12-18T02:52:06.992969Z",
  "requiresTranslation": false
}
```

---

#### Get Tag by Slug

```bash
GET /tags/slug/{slug}
```

Retrieves a specific tag by its URL-friendly slug identifier.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `slug` | string | Yes | URL-friendly tag slug |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `include_template` | boolean | Include template data |

**Response:** Same as [Get Tag by ID](#get-tag-by-id)

**Example Request:**
```bash
curl "https://gamma-api.polymarket.com/tags/slug/trump"
```

**Example Response:**
```json
{
  "id": "126",
  "label": "Trump",
  "slug": "trump",
  "forceShow": false,
  "publishedAt": "2023-11-02 21:23:16.384+00",
  "updatedBy": 15,
  "createdAt": "2023-11-02T21:23:16.39Z",
  "updatedAt": "2025-12-18T02:52:06.992969Z"
}
```

> **Tip:** Use slugs when you have the URL-friendly identifier (e.g., from a Polymarket URL), and IDs when working with tag references from market/event objects.

---

#### Get Sports Metadata

```bash
GET /sports
```

Returns sports-specific metadata including tag IDs, images, resolution sources.

**Example - Get NBA markets:**
```bash
# First get sports to find NBA tag_id, then:
curl "https://gamma-api.polymarket.com/events?tag_id=100381&closed=false"
```

### Search

#### Search Markets

```bash
GET /search
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search term |

---

## Sports API - Teams & Leagues

Base URL: `https://gamma-api.polymarket.com`

Polymarket provides dedicated endpoints for sports betting data including leagues, teams, and metadata. These endpoints power sports market discovery and categorization across the platform.

### Get Sports Metadata

```bash
GET /sports
```

Retrieves metadata for various sports including images, resolution sources, ordering preferences, tags, and series information. This endpoint provides comprehensive sport configuration data used throughout the platform.

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique sport identifier |
| `sport` | string | The sport identifier or abbreviation (e.g., `nfl`, `nba`, `epl`) |
| `image` | string (uri) | URL to the sport's logo or image asset |
| `resolution` | string (uri) | URL to the official resolution source for the sport (e.g., league website) |
| `ordering` | string | Preferred ordering for sport display, typically `"home"` or `"away"` |
| `tags` | string | Comma-separated list of tag IDs associated with the sport for categorization and filtering |
| `series` | string | Series identifier linking the sport to a specific tournament or season series |
| `createdAt` | string (datetime) | Timestamp when the sport was added |

**Example Request:**
```bash
curl "https://gamma-api.polymarket.com/sports"
```

**Example Response:**
```json
{
  "id": 1,
  "sport": "ncaab",
  "image": "https://polymarket-upload.s3.us-east-2.amazonaws.com/marchmadness.jpeg",
  "resolution": "https://www.ncaa.com/march-madness-live/bracket",
  "ordering": "home",
  "tags": "1,100149,100639",
  "series": "39",
  "createdAt": "2025-11-05T19:27:45.399303Z"
}
```

**Popular Sports/Leagues:**

| Code | League | Tags |
|------|--------|------|
| `nfl` | NFL | `1,450,100639` |
| `nba` | NBA | `1,745,100639` |
| `mlb` | MLB | `1,100639,100381` |
| `nhl` | NHL | `1,899,100639` |
| `epl` | English Premier League | `1,82,306,100639,100350` |
| `ucl` | UEFA Champions League | `1,100977,100639,1234,1003` |
| `mma` | MMA/UFC | `1,100639` |
| `ncaab` | NCAA Basketball | `1,100149,100639` |
| `lol` | League of Legends | `1,64,65,100639` |
| `cs2` | Counter-Strike 2 | `1,64,100780,100639` |
| `val` | Valorant | `1,64,101672,100639` |
| `atp` | ATP Tennis | `1,864,100639,101232` |
| `wta` | WTA Tennis | `1,864,100639,102123` |

---

### Get Sports Market Types

```bash
GET /sports/market-types
```

Returns a list of all valid sports market types available on the platform. Use these values when filtering markets by the `sportsMarketTypes` parameter.

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `marketTypes` | string[] | List of all valid sports market types |

**Example Request:**
```bash
curl "https://gamma-api.polymarket.com/sports/market-types"
```

**Available Market Types by Category:**

| Category | Market Types |
|----------|--------------|
| **General** | `moneyline`, `spreads`, `totals`, `parlays`, `double_chance`, `correct_score` |
| **Baseball** | `nrfi` (No Run First Inning) |
| **Soccer** | `total_goals`, `both_teams_to_score`, `match_handicap` |
| **Team Totals** | `team_totals`, `team_totals_home`, `team_totals_away` |
| **First Half** | `first_half_moneyline`, `first_half_spreads`, `first_half_totals` |
| **Football Props** | `anytime_touchdowns`, `first_touchdowns`, `two_plus_touchdowns`, `passing_yards`, `passing_touchdowns`, `receiving_yards`, `receptions`, `rushing_yards` |
| **Basketball** | `points`, `rebounds`, `assists`, `assists_points_rebounds`, `threes`, `double_doubles` |
| **Tennis** | `total_games`, `tennis_first_set_totals`, `tennis_match_totals`, `tennis_set_handicap`, `tennis_first_set_winner`, `tennis_set_totals` |
| **Esports (MOBA)** | `moba_first_blood`, `moba_first_tower`, `moba_first_dragon`, `moba_total_kills` |
| **Esports (FPS)** | `shooter_rounds_total`, `shooter_round_handicap`, `shooter_first_pistol_round`, `shooter_second_pistol_round`, `map_handicap`, `map_participant_win_total`, `map_participant_win_one` |
| **UFC/MMA** | `ufc_go_the_distance`, `ufc_method_of_victory` |
| **Cricket** | `cricket_toss_winner`, `cricket_completed_match`, `cricket_toss_match_double`, `cricket_most_sixes`, `cricket_team_top_batter`, `cricket_match_to_go_till` |
| **Other** | `child_moneyline` |

**Usage Example - Filter by Market Type:**
```bash
# Get all moneyline markets
curl "https://gamma-api.polymarket.com/markets?sportsMarketTypes=moneyline&closed=false&limit=20"

# Get spread and totals markets
curl "https://gamma-api.polymarket.com/markets?sportsMarketTypes=spreads,totals&closed=false&limit=20"
```

---

### List Teams

```bash
GET /teams
```

Returns all teams available for sports markets.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Results per page (default: varies) |
| `offset` | integer | No | Pagination offset |
| `order` | string | No | Comma-separated fields to order by |
| `ascending` | boolean | No | Sort direction |
| `league` | string[] | No | Filter by league code(s) |
| `name` | string[] | No | Filter by team name(s) |
| `abbreviation` | string[] | No | Filter by abbreviation(s) |

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Team ID |
| `name` | string | Full team name |
| `league` | string | League code (e.g., `nfl`, `nba`) |
| `record` | string | Current season record |
| `logo` | string | Team logo URL |
| `abbreviation` | string | Short code (e.g., `NYG`, `LAL`) |
| `alias` | string | Alternate name |

**Examples:**
```bash
# Get all teams
curl "https://gamma-api.polymarket.com/teams?limit=100"

# Filter by league
curl "https://gamma-api.polymarket.com/teams?league=nfl&limit=50"

# Search by name
curl "https://gamma-api.polymarket.com/teams?name=Lakers"
```

**Sample Teams by League:**

| League | Teams | Examples |
|--------|-------|----------|
| NFL | 32+ | `NYG` Giants, `LAL` Rams, `KC` Chiefs |
| NBA | 30+ | `LAL` Lakers, `BOS` Celtics, `GSW` Warriors |
| MLB | 30+ | `NYY` Yankees, `LAD` Dodgers, `BOS` Red Sox |
| EPL | 20+ | `MUN` Man United, `LIV` Liverpool, `CHE` Chelsea |
| CS2 | 60+ | `faze` FaZe, `navi` NAVI, `vit` Vitality |
| LoL | 60+ | `t1` T1, `g2` G2, `fnc` Fnatic |

---

### Using Sports Data for Market Filtering

Combine sports/teams data with the events endpoint:

```bash
# Get NFL markets using tag from /sports
curl "https://gamma-api.polymarket.com/events?tag_id=450&closed=false&limit=20"

# Get NBA markets
curl "https://gamma-api.polymarket.com/events?tag_id=745&closed=false&limit=20"

# Get esports (general tag 64)
curl "https://gamma-api.polymarket.com/events?tag_id=64&closed=false&limit=20"
```

---

## CLOB API - Trading

Base URL: `https://clob.polymarket.com`

The Central Limit Order Book (CLOB) handles all trading operations.

### Authentication

Most CLOB endpoints require API authentication. Generate credentials using the Python client:

```python
from py_clob_client.client import ClobClient

client = ClobClient("https://clob.polymarket.com", key=PRIVATE_KEY, chain_id=137)
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)
```

### Market Data (No Auth Required)

#### Get Order Book

```bash
GET /book
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token_id` | string | Yes | CLOB token ID |

**Response:**
```json
{
  "market": "0x...",
  "asset_id": "101669...",
  "bids": [{"price": "0.50", "size": "1000"}, ...],
  "asks": [{"price": "0.51", "size": "500"}, ...]
}
```

#### Get Midpoint Price

```bash
GET /midpoint
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token_id` | string | Yes | CLOB token ID |

#### Get Price

```bash
GET /price
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token_id` | string | Yes | CLOB token ID |
| `side` | string | Yes | `BUY` or `SELL` |

#### Get Spread

```bash
GET /spread
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token_id` | string | Yes | CLOB token ID |

#### Get Last Trade Price

```bash
GET /last-trade-price
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token_id` | string | Yes | CLOB token ID |

### Markets

#### Get All CLOB Markets

```bash
GET /markets
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `next_cursor` | string | Pagination cursor |

#### Get Simplified Markets

```bash
GET /simplified-markets
```

Returns markets with minimal data for quick overview.

#### Get Sampling Markets

```bash
GET /sampling-markets
GET /sampling-simplified-markets
```

Returns a curated sample of active markets.

#### Get Market by Condition ID

```bash
GET /markets/{condition_id}
```

### Order Management (Auth Required)

#### Place Order

```bash
POST /order
```

**Body:**
```json
{
  "order": {
    "salt": "random_nonce",
    "maker": "0xYourAddress",
    "signer": "0xSignerAddress",
    "taker": "0x0000000000000000000000000000000000000000",
    "tokenId": "101669...",
    "makerAmount": "1000000",
    "takerAmount": "500000",
    "expiration": "0",
    "nonce": "0",
    "feeRateBps": "0",
    "side": 0,
    "signatureType": 0
  },
  "signature": "0x...",
  "orderType": "GTC"
}
```

**Order Types:**
- `GTC` - Good Till Cancelled
- `GTD` - Good Till Date
- `FOK` - Fill or Kill

#### Cancel Order

```bash
DELETE /order/{order_id}
```

#### Cancel All Orders

```bash
DELETE /orders
```

#### Get Open Orders

```bash
GET /orders
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `market` | string | Filter by market |
| `asset_id` | string | Filter by asset |

#### Get Order by ID

```bash
GET /order/{order_id}
```

### Trades

#### Get Trades

```bash
GET /trades
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `market` | string | Filter by market |
| `maker_address` | string | Filter by maker |

### Historical Data

#### Get Timeseries

```bash
GET /timeseries
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `token_id` | string | CLOB token ID |
| `interval` | string | Time interval |

### Health

#### Health Check

```bash
GET /
```

#### Server Time

```bash
GET /time
```

---

## Data API - User Data

Base URL: `https://data-api.polymarket.com`

Provides user holdings and on-chain activity data.

### User Holdings

```bash
GET /holdings/{address}
```

### User Activity

```bash
GET /activity/{address}
```

---

## WebSocket API - Real-time Data

### CLOB WebSocket

URL: `wss://ws-subscriptions-clob.polymarket.com/ws/`

#### Market Channel

Subscribe to real-time order book updates.

```javascript
// Subscribe
{
  "type": "subscribe",
  "channel": "market",
  "market": "condition_id_here"
}
```

**Events:**
- `book_update` - Order book changes
- `trade` - New trade executed
- `price_change` - Price changed

#### User Channel (Authenticated)

Subscribe to your order/trade updates.

```javascript
// Subscribe (requires auth headers)
{
  "type": "subscribe",
  "channel": "user"
}
```

**Events:**
- `order_placed` - Your order was placed
- `order_filled` - Your order was filled
- `order_cancelled` - Your order was cancelled

### Real-Time Data Stream (RTDS)

URL: `wss://ws-live-data.polymarket.com`

#### Crypto Prices

Real-time crypto price updates.

#### Comments

Real-time market comments.

---

## Python Client Methods

### py_clob_client (Official)

```python
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType

# Initialize
client = ClobClient(
    host="https://clob.polymarket.com",
    key=PRIVATE_KEY,
    chain_id=137  # Polygon
)

# Set credentials
creds = client.create_or_derive_api_creds()
client.set_api_creds(creds)
```

#### Market Data Methods

| Method | Description |
|--------|-------------|
| `client.get_markets()` | Get all CLOB markets |
| `client.get_simplified_markets()` | Get simplified market list |
| `client.get_sampling_markets()` | Get sample of active markets |
| `client.get_sampling_simplified_markets()` | Get simplified sample |
| `client.get_market(condition_id)` | Get specific market |
| `client.get_order_book(token_id)` | Get order book |
| `client.get_price(token_id)` | Get current price |
| `client.get_midpoint(token_id)` | Get midpoint price |
| `client.get_spread(token_id)` | Get bid-ask spread |
| `client.get_last_trade_price(token_id)` | Get last trade price |

#### Order Methods

| Method | Description |
|--------|-------------|
| `client.create_order(OrderArgs)` | Create signed order |
| `client.create_market_order(MarketOrderArgs)` | Create market order |
| `client.post_order(order)` | Submit order to CLOB |
| `client.create_and_post_order(OrderArgs)` | Create and submit in one call |
| `client.cancel(order_id)` | Cancel specific order |
| `client.cancel_all()` | Cancel all open orders |
| `client.get_order(order_id)` | Get order details |
| `client.get_orders()` | Get all open orders |
| `client.get_trades()` | Get trade history |

#### Example - Place Limit Order

```python
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

order_args = OrderArgs(
    price=0.50,           # 50 cents
    size=100,             # 100 shares
    side=BUY,             # or SELL
    token_id="101669..."  # CLOB token ID
)

response = client.create_and_post_order(order_args)
print(response)
```

#### Example - Place Market Order

```python
from py_clob_client.clob_types import MarketOrderArgs, OrderType

order_args = MarketOrderArgs(
    token_id="101669...",
    amount=50.0,  # $50 USDC worth
)

signed_order = client.create_market_order(order_args)
response = client.post_order(signed_order, orderType=OrderType.FOK)
```

### Polymarket Class (This Codebase)

Our `Polymarket` class wraps both Gamma and CLOB APIs:

```python
from agents.polymarket.polymarket import Polymarket

poly = Polymarket()
```

#### Market Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `poly.get_all_markets()` | `list[SimpleMarket]` | Get all Gamma markets |
| `poly.get_market(token_id)` | `SimpleMarket` | Get market by CLOB token |
| `poly.filter_markets_for_trading(markets)` | `list[SimpleMarket]` | Filter active markets |
| `poly.get_sampling_simplified_markets()` | `list[SimpleMarket]` | Get CLOB sampling markets |

#### Event Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `poly.get_all_events()` | `list[SimpleEvent]` | Get all Gamma events |
| `poly.filter_events_for_trading(events)` | `list[SimpleEvent]` | Filter tradeable events |
| `poly.get_all_tradeable_events()` | `list[SimpleEvent]` | Get active, non-restricted events |

#### Order Book Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `poly.get_orderbook(token_id)` | `OrderBookSummary` | Get full order book |
| `poly.get_orderbook_price(token_id)` | `float` | Get current price |

#### Trading Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `poly.execute_order(price, size, side, token_id)` | `str` | Place limit order |
| `poly.execute_market_order(market, amount)` | `str` | Place market order |
| `poly.build_order(token, amount, ...)` | `Order` | Build signed order |

#### Account Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `poly.get_usdc_balance()` | `float` | Get USDC balance |
| `poly.get_address_for_private_key()` | `str` | Get wallet address |

### GammaMarketClient (This Codebase)

Extended Gamma API wrapper:

```python
from agents.polymarket.gamma import GammaMarketClient

gamma = GammaMarketClient()
```

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `gamma.get_markets(params)` | `list[dict]` | Query markets with params |
| `gamma.get_events(params)` | `list[dict]` | Query events with params |
| `gamma.get_all_markets(limit)` | `list[dict]` | Get all markets |
| `gamma.get_all_events(limit)` | `list[dict]` | Get all events |
| `gamma.get_current_markets(limit)` | `list[dict]` | Get active markets |
| `gamma.get_current_events(limit)` | `list[dict]` | Get active events |
| `gamma.get_clob_tradable_markets(limit)` | `list[dict]` | Get CLOB-enabled markets |
| `gamma.get_market(market_id)` | `dict` | Get market by ID |
| `gamma.get_event(event_id)` | `dict` | Get event by ID |
| `gamma.get_all_current_markets(limit)` | `list[dict]` | Paginated all active markets |

---

## Rate Limits

| Endpoint Type | Rate Limit |
|---------------|------------|
| Public REST | 10 requests/second |
| Authenticated REST | 100 requests/second |
| WebSocket | 100 messages/second |

See [API Rate Limits](https://docs.polymarket.com/quickstart/introduction/rate-limits) for details.

---

## Best Practices

### ✅ Do

1. **Use slug method for individual markets** - Most performant for specific lookups
2. **Use tag filtering for categories** - Reduces API calls
3. **Use events endpoint for discovery** - Events contain their markets
4. **Always include `closed=false`** - Unless you need historical data
5. **Implement rate limiting** - Add delays between requests (0.2-0.5s)
6. **Paginate large requests** - Use `limit` + `offset` for full data

### ❌ Don't

1. Don't fetch all markets without filtering
2. Don't make rapid sequential API calls
3. Don't request closed markets unless needed
4. Don't ignore pagination for large datasets

---

## Market Data Structure

### Market Object (Gamma)

```json
{
  "id": 253123,
  "question": "Will Trump win the 2024 election?",
  "description": "...",
  "outcomes": ["Yes", "No"],
  "outcomePrices": ["0.55", "0.45"],
  "clobTokenIds": ["101669...", "202558..."],
  "active": true,
  "closed": false,
  "archived": false,
  "endDate": "2024-11-06T00:00:00Z",
  "volume": 50000000.0,
  "spread": 0.02,
  "rewardsMinSize": 5.0,
  "rewardsMaxSpread": 0.05
}
```

### Event Object (Gamma)

```json
{
  "id": 12345,
  "ticker": "2024-ELECTION",
  "slug": "2024-presidential-election",
  "title": "2024 Presidential Election",
  "description": "...",
  "active": true,
  "closed": false,
  "archived": false,
  "new": false,
  "featured": true,
  "restricted": false,
  "endDate": "2024-11-06T00:00:00Z",
  "markets": [{"id": 253123, ...}, ...]
}
```

### Order Book (CLOB)

```json
{
  "market": "0x26ee82bee2493a302d21283cb578f7e2fff2dd15743854f53034d12420863b55",
  "asset_id": "101669189743438912873361127612589311253202068943959811456820079057046819967115",
  "bids": [
    {"price": "0.50", "size": "10000"},
    {"price": "0.49", "size": "5000"}
  ],
  "asks": [
    {"price": "0.51", "size": "8000"},
    {"price": "0.52", "size": "3000"}
  ]
}
```

---

## Common Workflows

### 1. Find and Fetch a Specific Market

```python
# By slug (from Polymarket URL)
curl "https://gamma-api.polymarket.com/events/slug/fed-decision-in-october"

# By market ID
curl "https://gamma-api.polymarket.com/markets/253123"

# By CLOB token ID
curl "https://gamma-api.polymarket.com/markets?clob_token_ids=101669..."
```

### 2. Get Active Markets by Category

```python
# Get all sports tags
curl "https://gamma-api.polymarket.com/sports"

# Filter by tag
curl "https://gamma-api.polymarket.com/events?tag_id=100381&closed=false&limit=50"
```

### 3. Place a Trade (Python)

```python
from agents.polymarket.polymarket import Polymarket

poly = Polymarket()

# Get market
market = poly.get_market("101669...")

# Check price
price = poly.get_orderbook_price("101669...")

# Execute order
response = poly.execute_order(
    price=0.50,
    size=100,
    side="BUY",
    token_id="101669..."
)
```

### 4. Monitor Real-time Prices (WebSocket)

```javascript
const ws = new WebSocket('wss://ws-subscriptions-clob.polymarket.com/ws/');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'market',
    market: 'condition_id_here'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Update:', data);
};
```

---

## Additional Resources

- [Official Polymarket Docs](https://docs.polymarket.com)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client)
- [Exchange Contract Source](https://github.com/Polymarket/ctf-exchange)
- [Discord #devs Channel](https://discord.gg/polymarket)

