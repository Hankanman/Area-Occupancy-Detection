"""A small REST client for driving a running instance.

Everything the harness needs from a live Home Assistant is reachable over the
HTTP API: onboarding, states, services, and -- the reason this exists -- the
config, options and subentry flow managers. Driving flows over HTTP is the
only way to see what the frontend sees, which is where the failures unit
tests cannot reach live: a selector schema core rejects, a step id with no
handler behind it, a translation placeholder that never resolves.

Deliberately stdlib-only, so the harness has no dependency the repo does not
already have for other reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

#: Client id presented to the auth provider. Must be a URL.
CLIENT_ID = "http://localhost:8123/"

#: Instances only ever listen on loopback, so any ambient proxy settings are
#: wrong for them. Using an explicit opener keeps the harness working in
#: environments where everything else goes through a proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ApiError(RuntimeError):
    """An HTTP call to the instance failed.

    Attributes:
        status: HTTP status code, or 0 if the request never completed.
        body: Response body, truncated by the caller if needed.
    """

    def __init__(self, message: str, *, status: int = 0, body: str = "") -> None:
        """Initialise the error.

        Args:
            message: Human-readable summary.
            status: HTTP status code.
            body: Response body.
        """
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(slots=True)
class Client:
    """Authenticated HTTP client for one instance.

    Access tokens last half an hour, which is shorter than an afternoon of
    poking at an instance, so the refresh token is kept and used to mint a new
    one the first time a call comes back unauthorised.

    Attributes:
        base_url: Instance root, without a trailing slash.
        token: Bearer token, set once onboarding has run.
        refresh_token: Refresh token used to renew an expired bearer token.
        timeout: Per-request timeout in seconds.
    """

    base_url: str
    token: str | None = None
    refresh_token: str | None = None
    timeout: float = 30.0

    def request(
        self,
        method: str,
        path: str,
        payload: Any = None,
        *,
        form: dict[str, str] | None = None,
        authenticated: bool = True,
        allow_refresh: bool = True,
    ) -> Any:
        """Make a request and decode the JSON response.

        Args:
            method: HTTP method.
            path: Path beginning with ``/``.
            payload: Object to send as a JSON body.
            form: Form fields to send instead of a JSON body.
            authenticated: Whether to send the bearer token.
            allow_refresh: Whether a 401 may trigger a token refresh and one
                retry. Set false internally to stop a refresh loop.

        Returns:
            The decoded response, or ``None`` for an empty body.

        Raises:
            ApiError: If the request fails or returns a non-2xx status.
        """
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        body: bytes | None = None

        if form is not None:
            body = urllib.parse.urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"

        if authenticated and self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", "replace")
            if (
                err.code == 401
                and authenticated
                and allow_refresh
                and self.refresh_token
            ):
                self.refresh()
                return self.request(
                    method,
                    path,
                    payload,
                    form=form,
                    authenticated=authenticated,
                    allow_refresh=False,
                )
            raise ApiError(
                f"{method} {path} failed with {err.code}: {detail[:400]}",
                status=err.code,
                body=detail,
            ) from err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise ApiError(f"{method} {path} failed: {err}") from err

        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode("utf-8", "replace")

    def refresh(self) -> str:
        """Exchange the refresh token for a fresh access token.

        Returns:
            The new access token.

        Raises:
            ApiError: If there is no refresh token, or the exchange fails.
        """
        if not self.refresh_token:
            raise ApiError("no refresh token to renew the access token with")
        renewed = self.request(
            "POST",
            "/auth/token",
            form={
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": CLIENT_ID,
            },
            authenticated=False,
        )
        self.token = renewed["access_token"]
        return self.token

    def get(self, path: str) -> Any:
        """Make a GET request.

        Args:
            path: Path beginning with ``/``.

        Returns:
            The decoded response.
        """
        return self.request("GET", path)

    def post(self, path: str, payload: Any = None) -> Any:
        """Make a POST request with a JSON body.

        Args:
            path: Path beginning with ``/``.
            payload: Object to send.

        Returns:
            The decoded response.
        """
        return self.request("POST", path, payload)

    # -- onboarding -----------------------------------------------------

    def onboard(self, *, username: str, password: str, name: str = "Harness") -> str:
        """Run onboarding to completion and return an access token.

        The public onboarding API is used rather than writing the auth store
        by hand: it is a stable contract, and it leaves the instance in the
        state a real first run does.

        Args:
            username: Login username to create.
            password: Login password to create.
            name: Display name for the owner account.

        Returns:
            A bearer token for the new owner account. The refresh token is
            kept on the client.

        Raises:
            ApiError: If onboarding is already done, or any step fails.
        """
        created = self.request(
            "POST",
            "/api/onboarding/users",
            {
                "client_id": CLIENT_ID,
                "name": name,
                "username": username,
                "password": password,
                "language": "en",
            },
            authenticated=False,
        )
        token = self.request(
            "POST",
            "/auth/token",
            form={
                "grant_type": "authorization_code",
                "code": created["auth_code"],
                "client_id": CLIENT_ID,
            },
            authenticated=False,
        )
        self.token = token["access_token"]
        self.refresh_token = token.get("refresh_token")

        # The remaining steps only mark onboarding done. Core config is
        # already set from the generated YAML, and analytics stays off.
        self.post("/api/onboarding/core_config")
        self.post("/api/onboarding/analytics")
        self.post(
            "/api/onboarding/integration",
            {"client_id": CLIENT_ID, "redirect_uri": CLIENT_ID},
        )
        return self.token

    def onboarding_done(self) -> bool:
        """Whether the instance has finished onboarding.

        Returns:
            True when every onboarding step reports done. Home Assistant
            removes the onboarding views once it is finished, so a 404 is
            itself the answer.
        """
        try:
            steps = self.request("GET", "/api/onboarding", authenticated=False)
        except ApiError as err:
            if err.status == 404:
                return True
            raise
        return all(step.get("done") for step in steps or [])

    # -- state and services ---------------------------------------------

    def states(self) -> list[dict[str, Any]]:
        """Every state in the instance.

        Returns:
            The list of state objects.
        """
        return self.get("/api/states")

    def state(self, entity_id: str) -> dict[str, Any] | None:
        """One entity's state.

        Args:
            entity_id: Entity to read.

        Returns:
            The state object, or ``None`` if the entity does not exist.
        """
        try:
            return self.get(f"/api/states/{entity_id}")
        except ApiError as err:
            if err.status == 404:
                return None
            raise

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict[str, Any] | None = None,
        *,
        return_response: bool = False,
    ) -> Any:
        """Call a service.

        Args:
            domain: Service domain.
            service: Service name.
            data: Service data.
            return_response: Ask for the service's response. Services
                declared with ``SupportsResponse.ONLY`` reject a call that
                does not, so this is required for them rather than optional.

        Returns:
            The service response.
        """
        query = "?return_response" if return_response else ""
        return self.post(f"/api/services/{domain}/{service}{query}", data or {})

    def set_state(self, entity_id: str, state: str) -> Any:
        """Drive a mock sensor by setting its knob.

        Args:
            entity_id: The input helper to move.
            state: Target state -- ``on``/``off`` for booleans, an option for
                selects, a number for numbers.

        Returns:
            The service response.
        """
        domain = entity_id.split(".", 1)[0]
        if domain == "input_boolean":
            return self.call_service(
                "input_boolean",
                "turn_on" if state == "on" else "turn_off",
                {"entity_id": entity_id},
            )
        if domain == "input_select":
            return self.call_service(
                "input_select",
                "select_option",
                {"entity_id": entity_id, "option": state},
            )
        if domain == "input_number":
            return self.call_service(
                "input_number",
                "set_value",
                {"entity_id": entity_id, "value": float(state)},
            )
        raise ApiError(
            f"cannot drive {entity_id}: unsupported helper domain {domain!r}"
        )

    # -- config entries and flows ---------------------------------------

    def config_entries(self, domain: str | None = None) -> list[dict[str, Any]]:
        """List config entries.

        Args:
            domain: Restrict to one integration.

        Returns:
            The matching entries.
        """
        query = f"?domain={domain}" if domain else ""
        return self.get(f"/api/config/config_entries/entry{query}")

    def start_options_flow(self, entry_id: str) -> dict[str, Any]:
        """Open the options flow for an entry.

        Args:
            entry_id: The entry to configure.

        Returns:
            The first flow step.
        """
        return self.post(
            "/api/config/config_entries/options/flow", {"handler": entry_id}
        )

    def advance_options_flow(self, flow_id: str, user_input: Any) -> dict[str, Any]:
        """Submit one options-flow step.

        Args:
            flow_id: The in-progress flow.
            user_input: The step's input, or ``{"next_step_id": ...}`` for a
                menu.

        Returns:
            The next flow step.
        """
        return self.post(
            f"/api/config/config_entries/options/flow/{flow_id}", user_input
        )

    def start_subentry_flow(
        self, entry_id: str, subentry_type: str, *, subentry_id: str | None = None
    ) -> dict[str, Any]:
        """Open a subentry flow, optionally to reconfigure an existing one.

        Args:
            entry_id: The parent entry.
            subentry_type: Subentry type to add or reconfigure.
            subentry_id: An existing subentry to reconfigure, which switches
                the flow's source to reconfigure.

        Returns:
            The first flow step.
        """
        payload: dict[str, Any] = {"handler": [entry_id, subentry_type]}
        if subentry_id:
            payload["subentry_id"] = subentry_id
        return self.post("/api/config/config_entries/subentries/flow", payload)

    def advance_subentry_flow(self, flow_id: str, user_input: Any) -> dict[str, Any]:
        """Submit one subentry-flow step.

        Args:
            flow_id: The in-progress flow.
            user_input: The step's input.

        Returns:
            The next flow step.
        """
        return self.post(
            f"/api/config/config_entries/subentries/flow/{flow_id}", user_input
        )

    def abort_options_flow(self, flow_id: str) -> None:
        """Delete an in-progress options flow, ignoring an already-gone one.

        Args:
            flow_id: The flow to abandon.
        """
        try:
            self.request("DELETE", f"/api/config/config_entries/options/flow/{flow_id}")
        except ApiError as err:
            if err.status not in (404, 405):
                raise

    def abort_subentry_flow(self, flow_id: str) -> None:
        """Delete an in-progress subentry flow, ignoring an already-gone one.

        Args:
            flow_id: The flow to abandon.
        """
        try:
            self.request(
                "DELETE", f"/api/config/config_entries/subentries/flow/{flow_id}"
            )
        except ApiError as err:
            if err.status not in (404, 405):
                raise
