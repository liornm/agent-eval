from metrics.base import ComparisonExample, Metric

CODE_REUSE = Metric(
    name="Code Reuse",
    description=(
        "Whether the code changes leverage existing utilities, helpers, and abstractions in the "
        "codebase rather than reimplementing functionality that already exists elsewhere."
    ),
    required_artifacts=["git_diff"],
    examples=[
        ComparisonExample(
            task=(
                "fix bug in test client when using follow_redirects the session shows "
                "the state from the first request instead of the last one. think the "
                "contexts are being restored in the wrong order after redirects. update "
                "test_redirect_keep_session to verify the fix and add changelog entry"
            ),
            better="""\
diff --git a/src/flask/testing.py b/src/flask/testing.py
@@ -240,9 +240,9 @@ class FlaskClient(Client):
         response.json_module = self.application.json  # type: ignore[assignment]

         # Re-push contexts that were preserved during the request.
-        while self._new_contexts:
-            cm = self._new_contexts.pop()
+        for cm in self._new_contexts:
             self._context_stack.enter_context(cm)
+        self._new_contexts.clear()

         return response

diff --git a/tests/test_testing.py b/tests/test_testing.py
@@ -142,10 +142,16 @@ def test_redirect_keep_session(app, client, app_ctx):
     @app.route("/", methods=["GET", "POST"])
     def index():
         if flask.request.method == "POST":
-            return flask.redirect("/getsession")
-        flask.session["data"] = "foo"
+            flask.session["data"] = "from-post"
+            return flask.redirect("/setlater")
+        flask.session["data"] = "from-get"
         return "index"

+    @app.route("/setlater")
+    def set_later():
+        flask.session["data"] = "from-redirect"
+        return flask.redirect("/getsession")
+
-        assert flask.session.get("data") == "foo"
+        assert flask.session.get("data") == "from-redirect"

         rv = client.post("/", data={}, follow_redirects=True)
-        assert rv.data == b"foo"
+        assert rv.data == b"from-redirect\"""",
            worse="""\
diff --git a/src/flask/testing.py b/src/flask/testing.py
@@ -240,9 +240,20 @@ class FlaskClient(Client):
         response.json_module = self.application.json  # type: ignore[assignment]

         # Re-push contexts that were preserved during the request.
-        while self._new_contexts:
-            cm = self._new_contexts.pop()
-            self._context_stack.enter_context(cm)
+        self._entered_contexts = []
+        for cm in self._new_contexts:
+            try:
+                ctx = cm.__enter__()
+                self._entered_contexts.append((cm, ctx))
+            except Exception:
+                # Clean up already entered contexts on failure
+                for entered_cm, _ in reversed(self._entered_contexts):
+                    try:
+                        entered_cm.__exit__(None, None, None)
+                    except Exception:
+                        pass
+                raise
+        self._new_contexts.clear()

         return response""",
            explanation=(
                "The better solution uses the existing ExitStack (self._context_stack) "
                "which already handles entering contexts and proper cleanup on exceptions. "
                "The worse solution reimplements ExitStack's functionality manually with "
                "try/except, __enter__/__exit__ calls, and a new list to track contexts - "
                "duplicating logic that already exists in the standard library abstraction "
                "the class is already using."
            ),
        ),
    ],
)
