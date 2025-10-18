class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        @cache
        def dp(i, j):
            if i == len(s) and (j == len(p) or (
                len(set(p[j:])) == 1 and '*' in set(p[j:])
            )):
                return True
            if i == len(s) or j == len(p):
                return False
            if p[j] == '?':
                return dp(i+1, j+1)
            elif p[j] == '*':
                return dp(i+1, j) or dp(i+1, j+1) or dp(i, j+1)
            else:
                return s[i] == p[j] and dp(i+1, j+1)
        return dp(0, 0)
