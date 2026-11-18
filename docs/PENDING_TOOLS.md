# Pending MCP Tools

Tools the training corpus expects but FRIDAY does not register yet.
These calls are preserved verbatim during vocabulary migration so the
intent model stays future-proof. Implement them in the backend, or
delete the intent from the corpus and tests together.

Total calls waiting on implementation: **9001**

| Tool | Calls | Inferred arguments |
| :--- | ---: | :--- |
| `calculate` | 3515 | expression |
| `minimize_all_windows` | 1293 | (none) |
| `toggle_night_light` | 1157 | enabled |
| `take_note` | 1054 | text |
| `change_brightness` | 795 | delta |
| `restore_all_windows` | 491 | (none) |
| `toggle_bluetooth` | 259 | enabled |
| `decrease_brightness` | 189 | amount |
| `increase_brightness` | 80 | amount |
| `cancel_alarm` | 54 | (none) |
| `docker_action` | 18 | action, container |
| `hue_lights` | 18 | color, room, state |
| `obs_record` | 18 | action |
| `slack_dm` | 12 | recipient, text |
| `notion_create_task` | 12 | priority, title |
| `git_quick_commit` | 12 | message |
| `spotify_search` | 12 | query |
| `steam_action` | 12 | action, game |

## Details

### `calculate` - 3515 calls

Argument shapes:

- `['expression']` x3515
- values for `expression`: '18% of 237' x11, '18% of 47' x11, '18% of 477' x11, '18% of 368' x11

Example prompts:
- `calculate 450 * 18` -> `{"expression": "450 * 18"}`
- `calulate 55 times 12` -> `{"expression": "55 * 12"}`
- `calculate 255 - 22` -> `{"expression": "255 - 22"}`

### `minimize_all_windows` - 1293 calls

Argument shapes:

- `[]` x1293

Example prompts:
- `show desktop` -> `{}`
- `can u close browser tab and also show desktop` -> `{}`
- `what is 483 times 25?, then minimize all windows` -> `{}`

### `toggle_night_light` - 1157 calls

Argument shapes:

- `['enabled']` x1086
- `['enable']` x71
- values for `enabled`: True x811, False x275
- values for `enable`: True x41, False x30

Example prompts:
- `I'm going to sleep now` -> `{"enabled": true}`
- `bedtime mode` -> `{"enabled": true}`
- `I'm going to sleep now` -> `{"enabled": true}`

### `take_note` - 1054 calls

Argument shapes:

- `['text']` x1049
- `['content']` x5
- values for `text`: 'meeting with team tomorrow at 10am' x1041, 'call John at 5pm' x5, 'buy groceries tomorrow' x3
- values for `content`: 'buy groceries tomorrow' x5

Example prompts:
- `check system stats and cpu usage, take a note meeting with team tomorrow at 10am` -> `{"text": "meeting with team tomorrow at 10am"}`
- `take a note meeting with team tomorrow at 10am, then google github trending repositories` -> `{"text": "meeting with team tomorrow at 10am"}`
- `start Task Manager, take a note meeting with team tomorrow at 10am and also snap File Explorer to left on monitor 2` -> `{"text": "meeting with team tomorrow at 10am"}`

### `change_brightness` - 795 calls

Argument shapes:

- `['delta']` x795
- values for `delta`: -25 x262, 20 x256, -20 x245, -15 x16

Example prompts:
- `increase brightness by 10` -> `{"delta": 10}`
- `decrease brightness by 20` -> `{"delta": -20}`
- `decrease brightness by 20` -> `{"delta": -20}`

### `restore_all_windows` - 491 calls

Argument shapes:

- `[]` x491

Example prompts:
- `reopen closed tab then draft an email to sarah@example.com about Project Update, then restore all windows` -> `{}`
- `restore all windows` -> `{}`
- `restore all windows` -> `{}`

### `toggle_bluetooth` - 259 calls

Argument shapes:

- `['enabled']` x259
- values for `enabled`: True x243, False x16

Example prompts:
- `turn off bluetooth` -> `{"enabled": false}`
- `enable bluetooth` -> `{"enabled": true}`
- `enable bluetooth` -> `{"enabled": true}`

### `decrease_brightness` - 189 calls

Argument shapes:

- `['amount']` x189
- values for `amount`: 20 x105, 30 x42, 25 x42

Example prompts:
- `please my eyes are hurting, make the screen a bit dimmer` -> `{"amount": 20}`
- `hey friday, my eyes are hurting, make the screen a bit dimmer please` -> `{"amount": 20}`
- `listen, the monitor is burning my retinas, dim it down thanks` -> `{"amount": 30}`

### `increase_brightness` - 80 calls

Argument shapes:

- `['amount']` x80
- values for `amount`: 25 x61, 20 x19

Example prompts:
- `friday, it's too dark to read the screen, brighten it up` -> `{"amount": 25}`
- `hey friday please it's too dark to read the screen, brighten it up please` -> `{"amount": 25}`
- `hold on, sunlight is hitting my desk, make display brighter right now` -> `{"amount": 25}`

### `cancel_alarm` - 54 calls

Argument shapes:

- `[]` x54

Example prompts:
- `friday, silence the morning alarm` -> `{}`
- `silence the morning alarm please thanks` -> `{}`
- `can you kill that annoying alarm please thanks` -> `{}`

### `docker_action` - 18 calls

Argument shapes:

- `['action', 'container']` x18
- values for `action`: 'restart' x6, 'stop' x6, 'start' x6
- values for `container`: 'redis' x6, 'postgres' x6, 'web app' x6

Example prompts:
- `restart redis container` -> `{"action": "restart", "container": "redis"}`
- `stop postgres docker` -> `{"action": "stop", "container": "postgres"}`
- `restart redis container` -> `{"action": "restart", "container": "redis"}`

### `hue_lights` - 18 calls

Argument shapes:

- `['color', 'room', 'state']` x18
- values for `room`: 'living room' x6, 'bedroom' x6, 'office' x6
- values for `state`: 'on' x12, 'off' x6
- values for `color`: 'none' x6, 'blue' x6, 'warm white' x6

Example prompts:
- `turn off living room lights` -> `{"room": "living room", "state": "off", "color": "none"}`
- `turn bedroom lights blue` -> `{"room": "bedroom", "state": "on", "color": "blue"}`
- `make office lights warm white` -> `{"room": "office", "state": "on", "color": "warm white"}`

### `obs_record` - 18 calls

Argument shapes:

- `['action']` x18
- values for `action`: 'start' x6, 'stop' x6, 'pause' x6

Example prompts:
- `start recording my screen in obs` -> `{"action": "start"}`
- `start recording my screen in obs` -> `{"action": "start"}`
- `stop obs recording` -> `{"action": "stop"}`

### `slack_dm` - 12 calls

Argument shapes:

- `['recipient', 'text']` x12
- values for `recipient`: 'Sarah' x6, 'Alex' x6
- values for `text`: 'meeting at 2' x6, 'I am ready' x6

Example prompts:
- `send Slack message to Sarah saying meeting at 2` -> `{"recipient": "Sarah", "text": "meeting at 2"}`
- `send Slack message to Sarah saying meeting at 2` -> `{"recipient": "Sarah", "text": "meeting at 2"}`
- `send Slack message to Sarah saying meeting at 2` -> `{"recipient": "Sarah", "text": "meeting at 2"}`

### `notion_create_task` - 12 calls

Argument shapes:

- `['priority', 'title']` x12
- values for `title`: 'Buy Groceries' x6, 'Review PR' x6
- values for `priority`: 'normal' x6, 'high' x6

Example prompts:
- `add task Buy Groceries to Notion` -> `{"title": "Buy Groceries", "priority": "normal"}`
- `add task Buy Groceries to Notion` -> `{"title": "Buy Groceries", "priority": "normal"}`
- `add task Buy Groceries to Notion` -> `{"title": "Buy Groceries", "priority": "normal"}`

### `git_quick_commit` - 12 calls

Argument shapes:

- `['message']` x12
- values for `message`: 'fix audio bug' x6, 'updated router' x6

Example prompts:
- `commit saying fix audio bug` -> `{"message": "fix audio bug"}`
- `commit changes saying updated router` -> `{"message": "updated router"}`
- `commit saying fix audio bug` -> `{"message": "fix audio bug"}`

### `spotify_search` - 12 calls

Argument shapes:

- `['query']` x12
- values for `query`: 'Daft Punk' x6, 'Bohemian Rhapsody' x6

Example prompts:
- `search for Daft Punk on Spotify` -> `{"query": "Daft Punk"}`
- `search for Daft Punk on Spotify` -> `{"query": "Daft Punk"}`
- `find Bohemian Rhapsody on spotify` -> `{"query": "Bohemian Rhapsody"}`

### `steam_action` - 12 calls

Argument shapes:

- `['action', 'game']` x12
- values for `action`: 'pause_downloads' x6, 'launch' x6
- values for `game`: 'all' x6, 'Cyberpunk' x6

Example prompts:
- `pause downloads on Steam` -> `{"action": "pause_downloads", "game": "all"}`
- `launch Cyberpunk on Steam` -> `{"action": "launch", "game": "Cyberpunk"}`
- `pause downloads on Steam` -> `{"action": "pause_downloads", "game": "all"}`

