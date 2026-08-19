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

## Addendum: `/admin/reload-policy` (Phase 3, canary lifecycle)

Not part of the original Day-1 freeze above -- added when the drift-triggered canary
lifecycle (PROJECT_PLAN.md's "Policy swap on drift") was implemented. `POST
/admin/reload-policy` forces the service's MLflow Model Registry poll immediately
(normally it runs on a background timer every `CANARY_POLL_INTERVAL_SECONDS`) and
returns the current policy/canary state:
`{policy_name, policy_version, challenger_active, challenger_version}`. This is
operational/inspection tooling for the demo, not part of the user-facing recommendation
API -- the swap itself never requires calling it.
