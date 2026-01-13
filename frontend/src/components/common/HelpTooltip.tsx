import { HelpCircle } from 'lucide-react';
import { useState } from 'react';
import './HelpTooltip.css';

interface HelpTooltipProps {
  content: string;
  title?: string;
}

export const HelpTooltip = ({ content, title }: HelpTooltipProps) => {
  const [show, setShow] = useState(false);

  return (
    <div className="help-tooltip-container">
      <button
        className="help-tooltip-trigger"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onClick={(e) => {
          e.preventDefault();
          setShow(!show);
        }}
        type="button"
      >
        <HelpCircle size={16} />
      </button>
      {show && (
        <div className="help-tooltip-content">
          {title && <div className="help-tooltip-title">{title}</div>}
          <div className="help-tooltip-text">{content}</div>
        </div>
      )}
    </div>
  );
};
