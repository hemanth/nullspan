export interface FieldSpec {
  type: 'text' | 'number' | 'date' | 'time' | 'datetime' | 'duration' | 'list' | 'estimate' | 'forecast';
  instructions?: string;
  unit?: string;
  decimals?: number;
  integer?: boolean;
  items?: 'text' | 'number' | 'date';
  min?: number;
  max?: number;
  step?: number;
  scale?: 'linear' | 'log';
  horizon?: number;
  interval?: string;
  percentiles?: boolean;
}

export interface ExtractionResult {
  model: string;
  latency_ms: number;
  answers: Record<string, any>;
}

export interface NullspanSchemaExtractor {
  fields: Record<string, FieldSpec>;
  run(state: string): ExtractionResult;
  batch(states: string[]): ExtractionResult[];
}

export default function extract(state: string, fields: Record<string, FieldSpec>): ExtractionResult;
export default function extract(fields: Record<string, FieldSpec>): NullspanSchemaExtractor;
export { extract };
