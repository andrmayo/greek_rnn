import { useState, useEffect } from "react";
import { getDefaultModel, getModels, changeModel } from "../api";

export function ModelSelector() {
  const [models, setModels] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getModels(), getDefaultModel()])
      .then(([models, defaultModel]) => {
        setModels(models);
        setSelected(defaultModel);
      })
      .catch(console.error);
  }, []);

  async function handleSelect(model: string) {
    await changeModel(model);
    setSelected(model);
  }

  return (
    <ul className="flex flex-col">
      {models.map((model) => (
        <li key={model}>
          <button
            onClick={() => handleSelect(model)}
            className={`w-full text-left px-2 py-2 text-sm font-['Noto_Sans'] hover:bg-[var(--color-bg)] ${
              selected === model
                ? "text-[var(--color-brand)]"
                : "text-[var(--color-text-secondary)]"
            }`}
          >
            {model}
          </button>
        </li>
      ))}
    </ul>
  );
}
