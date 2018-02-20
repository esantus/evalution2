package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="definition")
public class Definition {
	private Long definition_id;
	private String definition_value;
	public Definition() {
		super();
		// TODO Auto-generated constructor stub
	}
	public Definition(Long definition_id, String definition_value) {
		super();
		this.definition_id = definition_id;
		this.definition_value = definition_value;
	}
	@Id
	@Column(name="definition_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getDefinition_id() {
		return definition_id;
	}
	public void setDefinition_id(Long definition_id) {
		this.definition_id = definition_id;
	}
	public String getDefinition_value() {
		return definition_value;
	}
	public void setDefinition_value(String definition_value) {
		this.definition_value = definition_value;
	}
	@Override
	public String toString() {
		return "Definition [definition_id=" + definition_id + ", definition_value=" + definition_value + "]";
	}
}
