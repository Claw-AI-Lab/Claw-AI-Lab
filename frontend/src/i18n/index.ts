import { createContext, useContext } from 'react';
import zh from './zh';
import en from './en';

export type Locale = 'zh' | 'en';

const MESSAGES: Record<Locale, Record<string, string>> = { zh, en };

export interface LocaleContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  t: (key: string, vars?: Record<string, string | number>) => string;
}

function makeT(locale: Locale) {
  const msgs = MESSAGES[locale];
  return (key: string, vars?: Record<string, string | number>): string => {
    let text = msgs[key] ?? MESSAGES.zh[key] ?? key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        text = text.replace(`{${k}}`, String(v));
      }
    }
    return text;
  };
}

export const LocaleContext = createContext<LocaleContextValue>({
  locale: 'en',
  setLocale: () => {},
  t: makeT('en'),
});

export function useLocale() {
  return useContext(LocaleContext);
}

export { makeT };
