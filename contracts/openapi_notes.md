# REST API (FROZEN, Day 1)

Source: `Claude_Code_Execution_Spec.docx` §3.4.

| Method & path    | Request                          | Response                              |
|-------------------|------------------------------------|------------------------------------------|
| `POST /recommend` | `user_id: str`                     | `{track_id: str}`                         |
| `POST /feedback`  | `user_id, track_id, action`        | `{status: "ok"}`                          |
| `GET /metrics`    | —                                   | running CTR, reward, regret               |
| `GET /health`     | —                                   | `{status: "ok"}`                          |

`action` (in `/feedback`) is one of: `completed`, `liked`, `playlist`, `skip`, `replay`
— see `kafka_topics.md` for the reward mapping.

Swagger docs at `/docs` must render exactly these four routes, no undocumented params
(Track C acceptance criterion).
