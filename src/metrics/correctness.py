from metrics.base import ComparisonExample, Metric

CORRECTNESS = Metric(
    name="Correctness",
    description=(
        "Whether the solution produces correct behavior: handles edge cases, avoids bugs, "
        "implements the correct logic, and satisfies all explicit and implicit requirements."
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

         return response""",
            worse="""\
diff --git a/src/flask/testing.py b/src/flask/testing.py
@@ -240,9 +240,9 @@ class FlaskClient(Client):
         response.json_module = self.application.json  # type: ignore[assignment]

         # Re-push contexts that were preserved during the request.
-        while self._new_contexts:
-            cm = self._new_contexts.pop()
-            self._context_stack.enter_context(cm)
+        # Clear all contexts to avoid stale state
+        self._new_contexts.clear()

         return response""",
            explanation=(
                "The bug is that .pop() returns contexts in reverse (LIFO) order, so the "
                "session reflects the first request instead of the last. The better solution "
                "iterates in insertion order with 'for cm in self._new_contexts', preserving "
                "the final redirect's session. The worse solution clears all contexts entirely, "
                "breaking session preservation after redirects."
            ),
        ),
    ],
)
