interface ReconstructionDisplayProps {
  text: string;
  lacunaMask: boolean[];
}

export function ReconstructionDisplay({
  text,
  lacunaMask,
}: ReconstructionDisplayProps) {
  const chars = Array.from(text);

  return (
    <p className="font-['Noto_Serif'] text-base leading-relaxed">
      {chars.map((char, i) =>
        lacunaMask[i] ? (
          <span
            key={i}
            className="text-[var(--color-brand)] border-b border-[var(--color-brand)]"
          >
            {char}
          </span>
        ) : (
          <span key={i}>{char}</span>
        ),
      )}
    </p>
  );
}
